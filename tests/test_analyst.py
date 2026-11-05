import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
import respx
from pydantic import HttpUrl

from techdebate.analyst import Analyst
from techdebate.config import Settings
from techdebate.errors import LLMError, PromptError, ToolError
from techdebate.llm import FakeLLM, LLMClient, LLMRequest, LLMResponse, ModelTier
from techdebate.models import Evidence, SpeakDecision
from techdebate.roles import ROLE_SPECS, RoleSpec
from techdebate.tools import ToolInput, ToolRegistry, default_tools, make_http_client


class _StubInput(ToolInput):
    query: str


def _item(name: str, query: str = "q") -> Evidence:
    return Evidence(
        source=name,
        query=query,
        url=HttpUrl(f"https://example.com/{name}"),
        title=name,
        summary=f"{name} summary",
    )


class _StubTool:
    description = "Finds things"
    input_model = _StubInput

    def __init__(self, name: str, run: Callable[[_StubInput], Awaitable[list[Evidence]]]) -> None:
        self.name = name
        self._run = run

    async def run(self, input: _StubInput) -> list[Evidence]:
        return await self._run(input)


def _finds(name: str) -> _StubTool:
    async def run(input: _StubInput) -> list[Evidence]:
        return [_item(name, input.query)]

    return _StubTool(name, run)


def _fails(name: str, error: Exception | None = None) -> _StubTool:
    async def run(input: _StubInput) -> list[Evidence]:
        raise error or ToolError(f"{name}: down")

    return _StubTool(name, run)


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def _analyst(*tools: _StubTool, llm: LLMClient | None = None, **settings: Any) -> Analyst:
    role = RoleSpec(
        name="developer",
        display_name="Developer",
        question="Are developers using it?",
        tool_names=tuple(tool.name for tool in tools),
        prompt_id="developer.v1",
    )
    return Analyst(role, llm or FakeLLM(), ToolRegistry(tools), _settings(**settings))


def _replies(text: str) -> FakeLLM:
    return FakeLLM(rules=[(lambda _: True, text)])


# Constructor


def test_a_bad_prompt_id_fails_when_the_analyst_is_created() -> None:
    role = RoleSpec(name="x", display_name="X", question="Why?", tool_names=(), prompt_id="malformed")
    with pytest.raises(PromptError, match="invalid prompt ID"):
        Analyst(role, FakeLLM(), ToolRegistry(), _settings())


# decide


def _decides(decision: SpeakDecision) -> FakeLLM:
    return FakeLLM(rules=[(lambda _: True, decision)])


async def test_decide_builds_the_request_from_the_role_and_the_settings() -> None:
    llm = _decides(SpeakDecision(speak=True, reply_to=None, reason="In my area."))

    await _analyst(llm=llm, decide_max_tokens=321).decide("Rust")

    [request] = llm.calls
    assert request.tier is ModelTier.CHEAP
    assert request.max_tokens == 321
    assert request.prompt_id == "decide.v1"
    assert request.output_schema is SpeakDecision


async def test_decide_keeps_the_topic_out_of_the_system_prompt() -> None:
    llm = _decides(SpeakDecision(speak=False, reply_to=None, reason="Not my area."))

    await _analyst(llm=llm).decide("Rust")

    [request] = llm.calls
    assert "You are the Developer" in request.system
    assert "Are developers using it?" in request.system
    assert "Rust" not in request.system
    [message] = request.messages
    assert message.role == "user"
    assert message.content == "Topic: Rust"


async def test_decide_returns_the_decision() -> None:
    decision = SpeakDecision(speak=False, reply_to=None, reason="Not my area.")

    assert await _analyst(llm=_decides(decision)).decide("Rust") == decision


async def test_decide_makes_no_tool_call() -> None:
    tool = _fails("a", KeyError("must not be called"))
    llm = _decides(SpeakDecision(speak=True, reply_to=None, reason="In my area."))

    await _analyst(tool, llm=llm).decide("Rust")


async def test_decide_raises_when_the_reply_does_not_fit_the_schema() -> None:
    with pytest.raises(LLMError, match="SpeakDecision"):
        await _analyst(llm=_replies('{"speak": true, "reply_to": 1, "reason": "No turn 1 exists."}')).decide("Rust")


# gather


async def test_gather_calls_every_tool_with_the_topic_as_the_query() -> None:
    evidence = await _analyst(_finds("a"), _finds("b")).gather("rust")

    assert [e.query for e in evidence] == ["rust", "rust"]


async def test_gather_runs_the_tools_concurrently() -> None:
    barrier = asyncio.Barrier(2)  # only passes once both tools are running at the same time

    async def run(input: _StubInput) -> list[Evidence]:
        await barrier.wait()
        return []

    async with asyncio.timeout(1):
        await _analyst(_StubTool("a", run), _StubTool("b", run)).gather("rust")


async def test_gather_keeps_the_role_tool_order_whichever_finishes_first() -> None:
    finish_first = asyncio.Event()

    async def slow(input: _StubInput) -> list[Evidence]:
        await finish_first.wait()
        return [_item("slow")]

    async def fast(input: _StubInput) -> list[Evidence]:
        finish_first.set()
        return [_item("fast")]

    evidence = await _analyst(_StubTool("slow", slow), _StubTool("fast", fast)).gather("rust")

    assert [e.source for e in evidence] == ["slow", "fast"]


async def test_gather_skips_a_failed_tool_and_logs_it(caplog: pytest.LogCaptureFixture) -> None:
    evidence = await _analyst(_finds("a"), _fails("b"), _finds("c")).gather("rust")

    assert [e.source for e in evidence] == ["a", "c"]
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "tool failed role=developer tool=b error=b: down"


async def test_gather_raises_with_each_error_when_every_tool_fails() -> None:
    with pytest.raises(ToolError, match="developer: every tool failed: a: down; b: down"):
        await _analyst(_fails("a"), _fails("b")).gather("rust")


async def test_gather_with_tools_that_find_nothing_is_not_a_failure() -> None:
    async def nothing(input: _StubInput) -> list[Evidence]:
        return []

    assert await _analyst(_StubTool("a", nothing)).gather("rust") == []


async def test_gather_lets_a_bug_through() -> None:
    with pytest.raises(KeyError):
        await _analyst(_finds("a"), _fails("b", KeyError("bug"))).gather("rust")


# write


async def test_write_builds_the_request_from_the_role_and_the_settings() -> None:
    llm = _replies("Rust is thriving.")

    await _analyst(llm=llm, write_max_tokens=1234).write("Rust", [_item("a")])

    [request] = llm.calls
    assert request.tier is ModelTier.STRONG
    assert request.max_tokens == 1234
    assert request.prompt_id == "developer.v1"
    assert request.output_schema is None


async def test_write_keeps_the_evidence_out_of_the_system_prompt() -> None:
    llm = _replies("Rust is thriving.")

    await _analyst(llm=llm).write("Rust", [_item("a"), _item("b")])

    [request] = llm.calls
    assert "You are the Developer" in request.system
    assert "Are developers using it?" in request.system
    assert "Rust" not in request.system
    assert "a summary" not in request.system

    [message] = request.messages
    assert message.role == "user"
    assert message.content == (
        "Topic: Rust\n\nEvidence:\n\n"
        "[1] a\n    a summary\n    https://example.com/a\n\n"
        "[2] b\n    b summary\n    https://example.com/b"
    )


async def test_write_with_no_evidence_says_so() -> None:
    llm = _replies("Nothing to go on.")

    await _analyst(llm=llm).write("Rust", [])

    assert llm.calls[0].messages[0].content == "Topic: Rust\n\nEvidence:\n\nNone found."


async def test_write_returns_a_turn_with_the_text_and_the_cost() -> None:
    class _PricedLLM(FakeLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            response = await super().generate(request)
            return response.model_copy(update={"cost_usd": 0.25})

    llm = _PricedLLM(rules=[(lambda _: True, "See [the repo](https://example.com/a).")])

    turn = await _analyst(llm=llm).write("Rust", [_item("a")])

    assert turn.analyst == "developer"
    assert turn.text == "See [the repo](https://example.com/a)."
    assert turn.cost_usd == 0.25


@pytest.mark.parametrize("text", ["", "  \n "])
async def test_write_raises_on_blank_text(text: str) -> None:
    with pytest.raises(LLMError, match=r"developer: fake returned a blank turn"):
        await _analyst(llm=_replies(text)).write("Rust", [_item("a")])


async def test_write_lets_an_llm_error_through() -> None:
    with pytest.raises(LLMError, match="no scripted reply"):
        await _analyst(llm=FakeLLM()).write("Rust", [])


# The real Developer tools, on recorded responses


@pytest.mark.usefixtures("mock_github", "mock_pypi")
async def test_gather_and_write_with_the_real_developer_tools() -> None:
    llm = _replies("Developers are building with it.")
    settings = _settings()

    async with make_http_client(settings) as client:
        analyst = Analyst(ROLE_SPECS[0], llm, default_tools(settings, client), settings)
        evidence = await analyst.gather("httpx")
    turn = await analyst.write("httpx", evidence)

    assert {e.source for e in evidence} == {"github_search", "github_repo", "pypi"}
    assert [e.source for e in evidence] == sorted((e.source for e in evidence), key=ROLE_SPECS[0].tool_names.index)
    content = llm.calls[0].messages[0].content
    assert all(str(e.url) in content for e in evidence)
    assert turn.text == "Developers are building with it."


@pytest.mark.usefixtures("mock_github")
async def test_real_tools_carry_on_when_pypi_does_not_know_the_topic(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host="pypi.org", path="/pypi/httpx/json").respond(404)
    settings = _settings()

    async with make_http_client(settings) as client:
        analyst = Analyst(ROLE_SPECS[0], FakeLLM(), default_tools(settings, client), settings)
        evidence = await analyst.gather("httpx")

    assert {e.source for e in evidence} == {"github_search", "github_repo"}
