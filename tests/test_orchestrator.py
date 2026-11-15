import logging
from typing import Any

import pytest
from pydantic import HttpUrl

from techdebate.config import Settings
from techdebate.errors import LLMError, PromptError, ToolError
from techdebate.llm import CostMeter, FakeLLM, LLMRequest, LLMResponse
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import Evidence, Message, ReplyTo, ResearchPlan, SpeakDecision, User, UserKind
from techdebate.orchestrator import check_post
from techdebate.roles import RoleRegistry, RoleSpec
from techdebate.tools import ToolInput, ToolRegistry

_RESEARCH = ResearchPlan(speak=True, queries={"stub": ["rust lang"]}, reason="In my area.")
_SKIP = ResearchPlan(speak=False, queries={"stub": []}, reason="Not my area.")
_COMMENT = SpeakDecision(reason="The data is new.", reply_to=[ReplyTo(place=0, point="Downloads doubled.")])
_PASS = SpeakDecision(reason="Nothing new.", reply_to=[])
_PRICES = {"research.v1": 0.01, "decide.v1": 0.02, "developer.v1": 0.25}


class _StubInput(ToolInput):
    query: str


class _StubTool:
    description = "Finds things"
    input_model = _StubInput

    def __init__(self, name: str = "stub", error: ToolError | None = None) -> None:
        self.name = name
        self.queries: list[str] = []
        self._error = error

    async def run(self, input: _StubInput) -> list[Evidence]:
        self.queries.append(input.query)
        if self._error:
            raise self._error
        return [
            Evidence(
                source=self.name, query=input.query, url=HttpUrl("https://example.com/x"), title="x", summary="found"
            )
        ]


class _PricedLLM(FakeLLM):
    """Each call costs a set amount, by prompt."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await super().generate(request)
        return response.model_copy(update={"cost_usd": _PRICES[request.prompt_id or ""]})


_ME = User(username="me", kind=UserKind.REAL)
_POST = Message(author_id=_ME.id, text="Rust")
_OVERVIEW = "Post by me:\nRust\n\n  No comments yet."


def _role(name: str = "developer", prompt_id: str = "developer.v1") -> RoleSpec:
    return RoleSpec(
        name=name, display_name=name.title(), question="Who uses it?", tool_names=("stub",), prompt_id=prompt_id
    )


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def _by(prompt_id: str, name: str | None) -> Predicate:
    def matches(request: LLMRequest) -> bool:
        return request.prompt_id == prompt_id and (name is None or f"You are the {name.title()}" in request.system)

    return matches


def _plans(name: str, plan: ResearchPlan) -> tuple[Predicate, Reply]:
    return _by("research.v1", name), plan


def _decides(name: str, decision: SpeakDecision) -> tuple[Predicate, Reply]:
    return _by("decide.v1", name), decision


def _writes(text: str = "Developers use it.", name: str | None = None) -> tuple[Predicate, Reply]:
    return _by("developer.v1", name), text


def _prompt_ids(llm: FakeLLM) -> list[str | None]:
    return [request.prompt_id for request in llm.calls]


def _agents(*roles: RoleSpec) -> dict[str, User]:
    return {role.name: User(username=role.name, kind=UserKind.AGENT) for role in roles}


async def _run(
    llm: FakeLLM | CostMeter, *roles: RoleSpec, tool: _StubTool | None = None, agents: dict[str, User] | None = None
) -> list[Message]:
    roles = roles or (_role(),)
    return await check_post(
        _POST,
        author=_ME,
        agents=agents if agents is not None else _agents(*roles),
        llm=llm,
        roles=RoleRegistry(roles),
        tools=ToolRegistry([tool or _StubTool()]),
        settings=_settings(),
    )


# Research, decide and write


async def test_an_agent_researches_decides_and_comments_on_the_post() -> None:
    llm = FakeLLM(
        rules=[
            _plans("developer", _RESEARCH),
            _decides("developer", _COMMENT),
            _writes("Developers use [it](https://example.com/x)."),
        ]
    )
    tool = _StubTool()
    agents = _agents(_role())

    [comment] = await _run(llm, tool=tool, agents=agents)

    assert comment.text == "Developers use [it](https://example.com/x)."
    assert comment.author_id == agents["developer"].id
    assert (comment.post_id, comment.parent_id) == (_POST.id, _POST.id)  # a top-level comment
    assert tool.queries == ["rust lang"]  # the plan's query, not the post
    assert _prompt_ids(llm) == ["research.v1", "decide.v1", "developer.v1"]
    research, decide, write = llm.calls
    # Both cheap calls see the post alone, new to the agent, and the post is the only place open to it.
    assert research.messages[0].content == f"{_OVERVIEW}\n\nNew to you: the post."
    decide_block = "New to you: the post.\nYou can comment on: 0 (a top-level comment on the post)."
    report = decide.messages[0].content.removeprefix(f"{_OVERVIEW}\n\n{decide_block}\n\n")
    assert report.startswith("Tool results:\n\n")
    assert decide.output_schema is not None
    # The write call gets the post, the same tool report, and the point the decision gave for its place.
    assert write.messages[0].content == f"Post: Rust\n\n{report}\n\nYour point: Downloads doubled."
    assert "https://example.com/x" in report


async def test_a_pass_at_the_research_call_searches_and_writes_nothing() -> None:
    llm = FakeLLM(rules=[_plans("developer", _SKIP)])
    tool = _StubTool()

    assert await _run(llm, tool=tool) == []
    assert tool.queries == []
    assert _prompt_ids(llm) == ["research.v1"]


async def test_a_pass_at_the_decide_call_searches_but_writes_nothing() -> None:
    llm = FakeLLM(rules=[_plans("developer", _RESEARCH), _decides("developer", _PASS)])
    tool = _StubTool()

    assert await _run(llm, tool=tool) == []
    assert tool.queries == ["rust lang"]
    assert _prompt_ids(llm) == ["research.v1", "decide.v1"]


async def test_with_no_agents_there_are_no_comments() -> None:
    llm = FakeLLM()

    comments = await check_post(
        _POST, author=_ME, agents={}, llm=llm, roles=RoleRegistry(), tools=ToolRegistry(), settings=_settings()
    )

    assert comments == []
    assert llm.calls == []


async def test_two_agents_both_comment_on_one_post_in_registration_order() -> None:
    first, second = _role("first"), _role("second")
    llm = FakeLLM(
        rules=[
            _plans("first", _RESEARCH),
            _plans("second", _RESEARCH),
            _decides("first", _COMMENT),
            _decides("second", _COMMENT),
            _writes("First view.", "first"),
            _writes("Second view.", "second"),
        ]
    )
    agents = _agents(first, second)

    comments = await _run(llm, first, second, agents=agents)

    assert [(c.author_id, c.text) for c in comments] == [
        (agents["first"].id, "First view."),
        (agents["second"].id, "Second view."),
    ]
    assert all((c.post_id, c.parent_id) == (_POST.id, _POST.id) for c in comments)
    assert _prompt_ids(llm) == ["research.v1", "decide.v1", "developer.v1"] * 2


async def test_an_agent_after_a_pass_still_gets_its_look() -> None:
    first, second, third = _role("first"), _role("second"), _role("third")
    llm = FakeLLM(
        rules=[
            _plans("first", _SKIP),
            _plans("second", _RESEARCH),
            _plans("third", _RESEARCH),
            _decides("second", _PASS),
            _decides("third", _COMMENT),
            _writes(),
        ]
    )
    agents = _agents(first, second, third)

    comments = await _run(llm, first, second, third, agents=agents)

    assert [c.author_id for c in comments] == [agents["third"].id]


async def test_each_plan_and_decision_is_logged_at_info_with_its_reason(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM(
        rules=[
            _plans("first", _SKIP),
            _plans("second", _RESEARCH),
            _decides("second", _COMMENT),
            _writes(),
        ]
    )

    with caplog.at_level(logging.INFO, logger="techdebate.orchestrator"):
        await _run(llm, _role("first"), _role("second"))

    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, f"agent planned post_id={_POST.id} agent=first speak=False reason=Not my area."),
        (logging.INFO, f"agent planned post_id={_POST.id} agent=second speak=True reason=In my area."),
        (logging.INFO, f"agent decided post_id={_POST.id} agent=second reply_to=[0] reason=The data is new."),
    ]


# Errors


async def test_a_bad_prompt_id_fails_before_any_llm_call() -> None:
    llm = FakeLLM(rules=[_plans("first", _SKIP)])

    with pytest.raises(PromptError, match="invalid prompt ID"):
        await _run(llm, _role("first"), _role("second", prompt_id="malformed"))

    assert llm.calls == []


async def test_an_agent_can_still_comment_when_every_tool_fails() -> None:
    llm = FakeLLM(
        rules=[
            _plans("developer", _RESEARCH),
            _decides("developer", _COMMENT),
            _writes("It is widely used."),
        ]
    )

    [comment] = await _run(llm, tool=_StubTool(error=ToolError("stub: down")))

    assert comment.text == "It is widely used."
    for call in llm.calls[1:]:  # the decide and the write calls
        assert 'stub, query "rust lang": the tool failed' in call.messages[0].content


async def test_an_agent_that_researches_with_no_queries_decides_without_calling_a_tool() -> None:
    no_tools = ResearchPlan(speak=True, queries={"stub": []}, reason="Nothing to search.")
    llm = FakeLLM(
        rules=[
            _plans("developer", no_tools),
            _decides("developer", _COMMENT),
            _writes("It is widely used."),
        ]
    )
    tool = _StubTool()

    [comment] = await _run(llm, tool=tool)

    assert comment.text == "It is widely used."
    assert tool.queries == []
    assert llm.calls[1].messages[0].content.endswith("stub: not used")


async def test_an_llm_error_goes_up() -> None:
    with pytest.raises(LLMError, match="no scripted reply"):
        await _run(FakeLLM())


# Cost


async def test_a_meter_passed_in_counts_both_cheap_calls_and_the_write() -> None:
    meter = CostMeter(
        _PricedLLM(
            rules=[
                _plans("first", _SKIP),
                _plans("second", _RESEARCH),
                _decides("second", _COMMENT),
                _writes(),
            ]
        )
    )

    await _run(meter, _role("first"), _role("second"))

    assert meter.total_usd == pytest.approx(0.01 + 0.01 + 0.02 + 0.25)


async def test_a_meter_passed_in_counts_both_cheap_calls_when_the_agent_passes() -> None:
    meter = CostMeter(_PricedLLM(rules=[_plans("developer", _RESEARCH), _decides("developer", _PASS)]))

    assert await _run(meter) == []
    assert meter.total_usd == pytest.approx(0.03)
