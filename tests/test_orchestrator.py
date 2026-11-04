import logging
import random
from typing import Any

import pytest
from pydantic import HttpUrl

from techdebate.config import Settings
from techdebate.errors import LLMError, PromptError, ToolError
from techdebate.llm import FakeLLM, LLMRequest, LLMResponse
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import Debate, DebateStatus, Evidence, SpeakDecision
from techdebate.orchestrator import run_debate
from techdebate.roles import RoleRegistry, RoleSpec
from techdebate.tools import ToolInput, ToolRegistry

_SPEAK = SpeakDecision(speak=True, reply_to=None, reason="In my area.")
_PASS = SpeakDecision(speak=False, reply_to=None, reason="Not my area.")
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


def _role(name: str = "developer", prompt_id: str = "developer.v1") -> RoleSpec:
    return RoleSpec(
        name=name, display_name=name.title(), question="Who uses it?", tool_names=("stub",), prompt_id=prompt_id
    )


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **{"read_delays": False, **overrides})  # pyright: ignore[reportCallIssue]


def _decides(name: str, decision: SpeakDecision) -> tuple[Predicate, Reply]:
    def matches(request: LLMRequest) -> bool:
        return request.prompt_id == "decide.v1" and f"You are the {name.title()}" in request.system

    return matches, decision


def _writes(text: str = "Developers use it.") -> tuple[Predicate, Reply]:
    return (lambda request: request.prompt_id == "developer.v1"), text


def _prompt_ids(llm: FakeLLM) -> list[str | None]:
    return [request.prompt_id for request in llm.calls]


async def _run(llm: FakeLLM, *roles: RoleSpec, tool: _StubTool | None = None) -> Debate:
    return await run_debate(
        "Rust",
        llm=llm,
        roles=RoleRegistry(roles or [_role()]),
        tools=ToolRegistry([tool or _StubTool()]),
        settings=_settings(),
    )


# Speak and pass


async def test_a_role_that_speaks_gathers_and_writes_one_turn() -> None:
    llm = FakeLLM(rules=[_decides("developer", _SPEAK), _writes("Developers use [it](https://example.com/x).")])
    tool = _StubTool()

    debate = await _run(llm, tool=tool)

    assert debate.topic == "Rust"
    assert debate.status is DebateStatus.FINISHED
    [turn] = debate.turns
    assert turn.analyst == "developer"
    assert turn.text == "Developers use [it](https://example.com/x)."
    assert tool.queries == ["Rust"]
    assert _prompt_ids(llm) == ["decide.v1", "developer.v1"]
    assert "https://example.com/x" in llm.calls[1].messages[0].content


async def test_when_every_role_passes_the_debate_finishes_with_no_turns() -> None:
    llm = FakeLLM(rules=[_decides("developer", _PASS)])
    tool = _StubTool()

    debate = await _run(llm, tool=tool)

    assert debate.status is DebateStatus.FINISHED
    assert debate.turns == []
    assert tool.queries == []
    assert _prompt_ids(llm) == ["decide.v1"]


async def test_with_no_roles_the_debate_finishes_with_no_turns() -> None:
    llm = FakeLLM()

    debate = await run_debate("Rust", llm=llm, roles=RoleRegistry(), tools=ToolRegistry(), settings=_settings())

    assert debate.status is DebateStatus.FINISHED
    assert debate.turns == []
    assert llm.calls == []


async def test_the_first_role_to_speak_ends_the_run() -> None:
    llm = FakeLLM(rules=[_decides("first", _SPEAK), _writes()])

    debate = await _run(llm, _role("first"), _role("second"))

    assert [turn.analyst for turn in debate.turns] == ["first"]
    assert _prompt_ids(llm) == ["decide.v1", "developer.v1"]  # the second role never decided


async def test_the_next_role_gets_its_moment_after_a_pass() -> None:
    llm = FakeLLM(rules=[_decides("first", _PASS), _decides("second", _SPEAK), _writes()])

    debate = await _run(llm, _role("first"), _role("second"))

    assert [turn.analyst for turn in debate.turns] == ["second"]


async def test_each_decision_and_its_reason_is_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM(rules=[_decides("first", _PASS), _decides("second", _SPEAK), _writes()])

    with caplog.at_level(logging.INFO, logger="techdebate.orchestrator"):
        await _run(llm, _role("first"), _role("second"))

    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, "role decided role=first speak=False reason=Not my area."),
        (logging.INFO, "role decided role=second speak=True reason=In my area."),
    ]


# Reading delays


async def test_each_role_waits_its_delay_before_it_decides() -> None:
    llm = FakeLLM(rules=[_decides("first", _PASS), _decides("second", _PASS)])
    waited: list[tuple[float, int]] = []

    async def sleep(seconds: float) -> None:
        waited.append((seconds, len(llm.calls)))

    await run_debate(
        "Rust",
        llm=llm,
        roles=RoleRegistry([_role("first"), _role("second")]),
        tools=ToolRegistry([_StubTool()]),
        settings=_settings(read_delays=True, read_delay_min_s=3, read_delay_max_s=10),
        rng=random.Random(7),
        sleep=sleep,
    )

    expected = random.Random(7)
    assert waited == [(expected.uniform(3, 10), 0), (expected.uniform(3, 10), 1)]
    assert all(3 <= seconds <= 10 for seconds, _ in waited)


async def test_read_delays_off_skips_the_waiting() -> None:
    waited: list[float] = []

    async def sleep(seconds: float) -> None:
        waited.append(seconds)

    await run_debate(
        "Rust",
        llm=FakeLLM(rules=[_decides("developer", _PASS)]),
        roles=RoleRegistry([_role()]),
        tools=ToolRegistry([_StubTool()]),
        settings=_settings(read_delays=False),
        sleep=sleep,
    )

    assert waited == []


# Errors


async def test_a_bad_prompt_id_fails_before_any_waiting_or_llm_call() -> None:
    llm = FakeLLM(rules=[_decides("first", _PASS)])
    waited: list[float] = []

    async def sleep(seconds: float) -> None:
        waited.append(seconds)

    with pytest.raises(PromptError, match="invalid prompt ID"):
        await run_debate(
            "Rust",
            llm=llm,
            roles=RoleRegistry([_role("first"), _role("second", prompt_id="malformed")]),
            tools=ToolRegistry([_StubTool()]),
            settings=_settings(read_delays=True),
            sleep=sleep,
        )

    assert waited == []
    assert llm.calls == []


async def test_a_tool_error_goes_up_when_every_tool_fails() -> None:
    llm = FakeLLM(rules=[_decides("developer", _SPEAK), _writes()])

    with pytest.raises(ToolError, match="every tool failed"):
        await _run(llm, tool=_StubTool(error=ToolError("stub: down")))

    assert _prompt_ids(llm) == ["decide.v1"]  # nothing spent on writing


async def test_an_llm_error_goes_up() -> None:
    with pytest.raises(LLMError, match="no scripted reply"):
        await _run(FakeLLM())


# Cost


async def test_the_total_cost_counts_the_decisions_and_the_write() -> None:
    llm = _PricedLLM(rules=[_decides("first", _PASS), _decides("second", _SPEAK), _writes()])

    debate = await _run(llm, _role("first"), _role("second"))

    assert debate.total_cost_usd == pytest.approx(0.27)
    assert debate.turns[0].cost_usd == 0.25  # a turn's cost is only the cost of writing it


async def test_the_total_cost_counts_the_decisions_when_everyone_passes() -> None:
    llm = _PricedLLM(rules=[_decides("developer", _PASS)])

    debate = await _run(llm)

    assert debate.turns == []
    assert debate.total_cost_usd == pytest.approx(0.01)
