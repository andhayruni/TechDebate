import logging
from typing import Any

import pytest
from pydantic import HttpUrl

from techdebate.config import Settings
from techdebate.errors import LLMError, PromptError, ToolError
from techdebate.llm import CostMeter, FakeLLM, LLMRequest, LLMResponse
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import Evidence, Message, SpeakDecision, User, UserKind
from techdebate.orchestrator import check_post
from techdebate.roles import RoleRegistry, RoleSpec
from techdebate.tools import ToolInput, ToolRegistry

_SPEAK = SpeakDecision(speak=True, reply_to=None, queries={"stub": ["rust lang"]}, reason="In my area.")
_PASS = SpeakDecision(speak=False, reply_to=None, queries={"stub": []}, reason="Not my area.")
_PRICES = {"decide.v1": 0.01, "developer.v1": 0.25}


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


def _role(name: str = "developer", prompt_id: str = "developer.v1") -> RoleSpec:
    return RoleSpec(
        name=name, display_name=name.title(), question="Who uses it?", tool_names=("stub",), prompt_id=prompt_id
    )


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def _decides(name: str, decision: SpeakDecision) -> tuple[Predicate, Reply]:
    def matches(request: LLMRequest) -> bool:
        return request.prompt_id == "decide.v1" and f"You are the {name.title()}" in request.system

    return matches, decision


def _writes(text: str = "Developers use it.", name: str | None = None) -> tuple[Predicate, Reply]:
    def matches(request: LLMRequest) -> bool:
        return request.prompt_id == "developer.v1" and (name is None or f"You are the {name.title()}" in request.system)

    return matches, text


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
        agents=agents if agents is not None else _agents(*roles),
        llm=llm,
        roles=RoleRegistry(roles),
        tools=ToolRegistry([tool or _StubTool()]),
        settings=_settings(),
    )


# Speak and pass


async def test_an_agent_that_speaks_gathers_and_comments_on_the_post() -> None:
    llm = FakeLLM(rules=[_decides("developer", _SPEAK), _writes("Developers use [it](https://example.com/x).")])
    tool = _StubTool()
    agents = _agents(_role())

    [comment] = await _run(llm, tool=tool, agents=agents)

    assert comment.text == "Developers use [it](https://example.com/x)."
    assert comment.author_id == agents["developer"].id
    assert (comment.post_id, comment.parent_id) == (_POST.id, _POST.id)  # a top-level comment
    assert tool.queries == ["rust lang"]  # the decision's query, not the post
    assert _prompt_ids(llm) == ["decide.v1", "developer.v1"]
    assert llm.calls[0].messages[0].content == "Topic: Rust"  # the post's text, until the prompts speak of posts
    assert "https://example.com/x" in llm.calls[1].messages[0].content


async def test_when_every_agent_passes_there_are_no_comments() -> None:
    llm = FakeLLM(rules=[_decides("developer", _PASS)])
    tool = _StubTool()

    comments = await _run(llm, tool=tool)

    assert comments == []
    assert tool.queries == []
    assert _prompt_ids(llm) == ["decide.v1"]


async def test_with_no_agents_there_are_no_comments() -> None:
    llm = FakeLLM()

    comments = await check_post(
        _POST, agents={}, llm=llm, roles=RoleRegistry(), tools=ToolRegistry(), settings=_settings()
    )

    assert comments == []
    assert llm.calls == []


async def test_two_agents_both_comment_on_one_post_in_registration_order() -> None:
    first, second = _role("first"), _role("second")
    llm = FakeLLM(
        rules=[
            _decides("first", _SPEAK),
            _decides("second", _SPEAK),
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
    assert _prompt_ids(llm) == ["decide.v1", "developer.v1", "decide.v1", "developer.v1"]


async def test_an_agent_after_a_pass_still_gets_its_look() -> None:
    llm = FakeLLM(rules=[_decides("first", _PASS), _decides("second", _SPEAK), _writes()])
    agents = _agents(_role("first"), _role("second"))

    comments = await _run(llm, _role("first"), _role("second"), agents=agents)

    assert [c.author_id for c in comments] == [agents["second"].id]


async def test_each_decision_and_its_reason_is_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM(rules=[_decides("first", _PASS), _decides("second", _SPEAK), _writes()])

    with caplog.at_level(logging.INFO, logger="techdebate.orchestrator"):
        await _run(llm, _role("first"), _role("second"))

    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, f"agent decided post_id={_POST.id} agent=first speak=False reason=Not my area."),
        (logging.INFO, f"agent decided post_id={_POST.id} agent=second speak=True reason=In my area."),
    ]


# Errors


async def test_a_bad_prompt_id_fails_before_any_llm_call() -> None:
    llm = FakeLLM(rules=[_decides("first", _PASS)])

    with pytest.raises(PromptError, match="invalid prompt ID"):
        await _run(llm, _role("first"), _role("second", prompt_id="malformed"))

    assert llm.calls == []


async def test_an_agent_still_comments_when_every_tool_fails() -> None:
    llm = FakeLLM(rules=[_decides("developer", _SPEAK), _writes("From general knowledge, it is used.")])

    [comment] = await _run(llm, tool=_StubTool(error=ToolError("stub: down")))

    assert comment.text == "From general knowledge, it is used."
    assert 'stub, query "rust lang": the tool failed' in llm.calls[1].messages[0].content


async def test_an_agent_that_speaks_with_no_queries_comments_without_calling_a_tool() -> None:
    no_tools = SpeakDecision(speak=True, reply_to=None, queries={"stub": []}, reason="Nothing to search.")
    llm = FakeLLM(rules=[_decides("developer", no_tools), _writes("From general knowledge, it is used.")])
    tool = _StubTool()

    [comment] = await _run(llm, tool=tool)

    assert comment.text == "From general knowledge, it is used."
    assert tool.queries == []
    assert llm.calls[1].messages[0].content.endswith("stub: not used")


async def test_an_llm_error_goes_up() -> None:
    with pytest.raises(LLMError, match="no scripted reply"):
        await _run(FakeLLM())


# Cost


async def test_a_meter_passed_in_counts_the_decisions_and_the_write() -> None:
    meter = CostMeter(_PricedLLM(rules=[_decides("first", _PASS), _decides("second", _SPEAK), _writes()]))

    await _run(meter, _role("first"), _role("second"))

    assert meter.total_usd == pytest.approx(0.27)


async def test_a_meter_passed_in_counts_the_decisions_when_everyone_passes() -> None:
    meter = CostMeter(_PricedLLM(rules=[_decides("developer", _PASS)]))

    assert await _run(meter) == []
    assert meter.total_usd == pytest.approx(0.01)
