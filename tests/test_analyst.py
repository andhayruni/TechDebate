import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import anthropic
import pytest
import respx
from pydantic import HttpUrl

from techdebate.analyst import Analyst, ToolResult
from techdebate.config import Settings
from techdebate.errors import LLMError, NotFoundError, PromptError, ToolError
from techdebate.llm import FakeLLM, LLMClient, ModelTier
from techdebate.models import Evidence, SpeakDecision
from techdebate.roles import ROLE_SPECS, RoleSpec
from techdebate.tools import ToolInput, ToolRegistry, default_tools, make_http_client


class _StubInput(ToolInput):
    query: str


def _found(name: str) -> ToolResult:
    return ToolResult(tool=name, query="q", evidence=[_item(name)])


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


_SPEAK = SpeakDecision(speak=True, reply_to=None, reason="In my area.")
_PASS = SpeakDecision(speak=False, reply_to=None, reason="Not my area.")


async def test_decide_builds_the_request_from_the_role_and_the_settings() -> None:
    llm = _decides(_SPEAK)

    await _analyst(llm=llm, decide_max_tokens=321).decide("Rust")

    [request] = llm.calls
    assert request.tier is ModelTier.CHEAP
    assert request.max_tokens == 321
    assert request.prompt_id == "decide.v1"


async def test_decide_keeps_the_topic_out_of_the_system_prompt() -> None:
    llm = _decides(_PASS)

    await _analyst(llm=llm).decide("Rust")

    [request] = llm.calls
    assert "You are the Developer" in request.system
    assert "Are developers using it?" in request.system
    assert "Rust" not in request.system
    [message] = request.messages
    assert message.role == "user"
    assert message.content == "Topic: Rust"


async def test_decide_tells_the_role_about_its_tools_and_the_cap() -> None:
    llm = _decides(_PASS.model_copy(update={"queries": {"a": []}}))

    await _analyst(_finds("a"), llm=llm, max_queries_per_tool=4).decide("Rust")

    system = llm.calls[0].system
    assert "- a: Finds things. Its query is: A search query." in system
    assert "up to 4 queries" in system


async def test_decide_returns_the_decision_with_its_queries_per_tool() -> None:
    decision = _SPEAK.model_copy(update={"queries": {"a": ["x", "y"], "b": []}})

    result = await _analyst(_finds("a"), _finds("b"), llm=_decides(decision)).decide("Rust")

    assert result == decision
    assert type(result) is SpeakDecision


async def test_the_schema_asks_for_every_tool_and_survives_the_provider_transform() -> None:
    llm = _decides(_SPEAK.model_copy(update={"queries": {"a": [], "b": []}}))

    await _analyst(_finds("a"), _finds("b"), llm=llm).decide("Rust")

    schema = llm.calls[0].output_schema
    assert schema is not None
    sent = anthropic.transform_schema(schema.model_json_schema())
    queries = sent["$defs"]["ToolQueries"]
    assert queries["required"] == ["a", "b"]
    assert queries["additionalProperties"] is False
    assert set(sent["required"]) == {"speak", "reply_to", "queries", "reason"}


@pytest.mark.parametrize(
    "queries",
    [
        {"a": ["x"]},  # a tool is missing
        {"a": ["x"], "b": [], "c": []},  # not one of the role's tools
    ],
)
async def test_decide_raises_when_the_queries_do_not_match_the_tools(queries: dict[str, list[str]]) -> None:
    llm = _decides(_SPEAK.model_copy(update={"queries": queries}))

    with pytest.raises(LLMError, match="SpeakDecision"):
        await _analyst(_finds("a"), _finds("b"), llm=llm).decide("Rust")


async def test_decide_makes_no_tool_call() -> None:
    tool = _fails("a", KeyError("must not be called"))
    llm = _decides(_SPEAK.model_copy(update={"queries": {"a": ["x"]}}))

    await _analyst(tool, llm=llm).decide("Rust")


async def test_decide_raises_when_the_reply_does_not_fit_the_schema() -> None:
    reply = '{"speak": true, "reply_to": 1, "queries": {}, "reason": "No turn 1 exists."}'
    with pytest.raises(LLMError, match="SpeakDecision"):
        await _analyst(llm=_replies(reply)).decide("Rust")


def test_a_role_with_a_missing_tool_fails_when_the_analyst_is_created() -> None:
    role = RoleSpec(name="x", display_name="X", question="Why?", tool_names=("nope",), prompt_id="developer.v1")
    with pytest.raises(ToolError, match="unknown tool: nope"):
        Analyst(role, FakeLLM(), ToolRegistry(), _settings())


# gather


async def test_gather_runs_each_query_on_its_tool() -> None:
    results = await _analyst(_finds("a"), _finds("b")).gather({"a": ["x", "y"], "b": ["z"]})

    assert [(r.tool, r.query, [e.query for e in r.evidence], r.failed) for r in results] == [
        ("a", "x", ["x"], False),
        ("a", "y", ["y"], False),
        ("b", "z", ["z"], False),
    ]


@pytest.mark.parametrize("queries", [{}, {"a": []}, {"a": ["", "   "]}])
async def test_gather_marks_a_tool_with_no_queries_as_not_used(queries: dict[str, list[str]]) -> None:
    tool = _fails("a", KeyError("must not be called"))

    assert await _analyst(tool).gather(queries) == [ToolResult(tool="a", query=None)]


async def test_gather_ignores_queries_for_tools_the_role_does_not_have() -> None:
    results = await _analyst(_finds("a")).gather({"a": ["x"], "other": ["y"]})

    assert [r.tool for r in results] == ["a"]


async def test_gather_strips_and_drops_blank_and_duplicate_queries_before_the_cap() -> None:
    analyst = _analyst(_finds("a"), max_queries_per_tool=2)

    results = await analyst.gather({"a": ["  ", " x ", "x", "y", "z"]})

    assert [r.query for r in results] == ["x", "y"]


async def test_gather_runs_the_queries_concurrently() -> None:
    barrier = asyncio.Barrier(3)  # only passes once all three queries are running at the same time

    async def run(input: _StubInput) -> list[Evidence]:
        await barrier.wait()
        return []

    async with asyncio.timeout(1):
        await _analyst(_StubTool("a", run), _StubTool("b", run)).gather({"a": ["x", "y"], "b": ["z"]})


async def test_gather_keeps_the_role_tool_order_whichever_finishes_first() -> None:
    finish_first = asyncio.Event()

    async def slow(input: _StubInput) -> list[Evidence]:
        await finish_first.wait()
        return [_item("slow")]

    async def fast(input: _StubInput) -> list[Evidence]:
        finish_first.set()
        return [_item("fast")]

    results = await _analyst(_StubTool("slow", slow), _StubTool("fast", fast)).gather({"slow": ["q"], "fast": ["q"]})

    assert [r.tool for r in results] == ["slow", "fast"]


async def test_gather_marks_a_failed_tool_and_logs_it_and_keeps_the_rest(caplog: pytest.LogCaptureFixture) -> None:
    results = await _analyst(_finds("a"), _fails("b"), _finds("c")).gather({"a": ["x"], "b": ["y"], "c": ["z"]})

    assert [r.failed for r in results] == [False, True, False]
    assert [len(r.evidence) for r in results] == [1, 0, 1]
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "tool failed role=developer tool=b query='y' error=b: down"


async def test_gather_does_not_raise_when_every_tool_fails() -> None:
    results = await _analyst(_fails("a"), _fails("b")).gather({"a": ["x"], "b": ["y"]})

    assert [r.failed for r in results] == [True, True]


async def test_gather_treats_not_found_as_no_information_not_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    results = await _analyst(_fails("a", NotFoundError("a: unknown x"))).gather({"a": ["x"]})

    assert results == [ToolResult(tool="a", query="x")]
    assert caplog.records == []  # below warning level


async def test_gather_with_a_tool_that_finds_nothing_is_not_a_failure() -> None:
    async def nothing(input: _StubInput) -> list[Evidence]:
        return []

    assert await _analyst(_StubTool("a", nothing)).gather({"a": ["x"]}) == [ToolResult(tool="a", query="x")]


async def test_gather_lets_a_bug_through() -> None:
    with pytest.raises(KeyError):
        await _analyst(_finds("a"), _fails("b", KeyError("bug"))).gather({"a": ["x"], "b": ["y"]})


# write


async def test_write_builds_the_request_from_the_role_and_the_settings() -> None:
    llm = _replies("Rust is thriving.")

    await _analyst(llm=llm, write_max_tokens=1234).write("Rust", [_found("a")])

    [request] = llm.calls
    assert request.tier is ModelTier.STRONG
    assert request.max_tokens == 1234
    assert request.prompt_id == "developer.v1"
    assert request.output_schema is None


async def test_write_keeps_the_evidence_out_of_the_system_prompt() -> None:
    llm = _replies("Rust is thriving.")

    await _analyst(llm=llm).write("Rust", [_found("a"), _found("b")])

    [request] = llm.calls
    assert "You are the Developer" in request.system
    assert "Are developers using it?" in request.system
    assert "Rust" not in request.system
    assert "a summary" not in request.system

    [message] = request.messages
    assert message.role == "user"
    assert message.content == (
        "Topic: Rust\n\nTool results:\n\n"
        'a, query "q":\n[1] a\n    a summary\n    https://example.com/a\n\n'
        'b, query "q":\n[1] b\n    b summary\n    https://example.com/b'
    )


async def test_write_reports_each_outcome_per_query() -> None:
    llm = _replies("Rust is thriving.")

    await _analyst(llm=llm).write(
        "Rust",
        [
            _found("a"),
            ToolResult(tool="a", query="other"),
            ToolResult(tool="b", query="x", failed=True),
            ToolResult(tool="c", query=None),
        ],
    )

    assert llm.calls[0].messages[0].content == (
        "Topic: Rust\n\nTool results:\n\n"
        'a, query "q":\n[1] a\n    a summary\n    https://example.com/a\n\n'
        'a, query "other": no information found\n\n'
        'b, query "x": the tool failed\n\n'
        "c: not used"
    )


async def test_write_still_runs_with_no_evidence_at_all() -> None:
    llm = _replies("From general knowledge, it is widely used.")

    text = await _analyst(llm=llm).write("Rust", [ToolResult(tool="a", query=None)])

    assert text == "From general knowledge, it is widely used."
    assert llm.calls[0].messages[0].content == "Topic: Rust\n\nTool results:\n\na: not used"


async def test_write_returns_the_text_of_the_comment_as_written() -> None:
    llm = _replies("See [the repo](https://example.com/a).")

    text = await _analyst(llm=llm).write("Rust", [_found("a")])

    assert text == "See [the repo](https://example.com/a)."


@pytest.mark.parametrize("text", ["", "  \n "])
async def test_write_raises_on_blank_text(text: str) -> None:
    with pytest.raises(LLMError, match=r"developer: fake returned a blank comment"):
        await _analyst(llm=_replies(text)).write("Rust", [_found("a")])


async def test_write_lets_an_llm_error_through() -> None:
    with pytest.raises(LLMError, match="no scripted reply"):
        await _analyst(llm=FakeLLM()).write("Rust", [])


# The real Developer tools, on recorded responses


_HTTPX = {"github_search": ["httpx"], "github_repo": ["httpx"], "pypi": ["httpx"]}


@pytest.mark.usefixtures("mock_github", "mock_pypi")
async def test_gather_and_write_with_the_real_developer_tools() -> None:
    llm = _replies("Developers are building with it.")
    settings = _settings()

    async with make_http_client(settings) as client:
        analyst = Analyst(ROLE_SPECS[0], llm, default_tools(settings, client), settings)
        results = await analyst.gather(_HTTPX)
    text = await analyst.write("httpx", results)

    assert [r.tool for r in results] == list(ROLE_SPECS[0].tool_names)
    content = llm.calls[0].messages[0].content
    assert all(str(e.url) in content for r in results for e in r.evidence)
    assert all(r.evidence for r in results)
    assert text == "Developers are building with it."


@pytest.mark.usefixtures("mock_github")
async def test_a_real_unknown_package_reads_as_no_information(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host="pypi.org", path="/pypi/httpx/json").respond(404)
    settings = _settings()

    async with make_http_client(settings) as client:
        analyst = Analyst(ROLE_SPECS[0], FakeLLM(), default_tools(settings, client), settings)
        results = await analyst.gather(_HTTPX)

    assert results[2] == ToolResult(tool="pypi", query="httpx")
    assert results[0].evidence
    assert results[1].evidence


@pytest.mark.usefixtures("mock_github")
async def test_a_real_outage_reads_as_a_failure(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host="pypi.org", path="/pypi/httpx/json").respond(500)
    settings = _settings()

    async with make_http_client(settings) as client:
        analyst = Analyst(ROLE_SPECS[0], FakeLLM(), default_tools(settings, client), settings)
        results = await analyst.gather(_HTTPX)

    assert results[2] == ToolResult(tool="pypi", query="httpx", failed=True)
