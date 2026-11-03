import argparse
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx
from pydantic import HttpUrl

from techdebate import cli
from techdebate.config import Settings
from techdebate.errors import LLMError, ToolError
from techdebate.llm import FakeLLM, LLMClient, LLMRequest, LLMResponse, Message, ModelTier
from techdebate.logging_setup import setup_logging
from techdebate.models import Evidence
from techdebate.prompts import load_prompt
from techdebate.tools import Tool, ToolInput, ToolRegistry


@pytest.mark.parametrize(
    ("argv", "verbose"),
    [
        (["run", "Rust"], False),
        (["-v", "run", "Rust"], True),
        (["run", "Rust", "-v"], True),
        (["ask", "What is Rust?", "-v"], True),
        (["evidence", "github_repo", "rust-lang/rust", "-v"], True),
    ],
)
def test_verbose_flag_before_or_after_subcommand(argv: list[str], verbose: bool) -> None:
    assert cli._build_parser().parse_args(argv).verbose is verbose  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(("verbose", "level"), [(True, logging.DEBUG), (False, logging.WARNING)])
def test_setup_logging_only_raises_our_level(verbose: bool, level: int) -> None:
    setup_logging(verbose)
    assert logging.getLogger("techdebate").level == level
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING


def test_techdebate_error_becomes_clean_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(args: argparse.Namespace) -> None:
        raise ToolError("github is down")

    monkeypatch.setattr(cli, "_run", fail)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "Rust"])

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "error: github is down" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("tier", list(ModelTier))
async def test_ask_builds_request_from_ask_prompt(tier: ModelTier) -> None:
    llm = FakeLLM(by_prompt={"What is Rust?": "A systems language."})

    response = await cli.ask(llm, "What is Rust?", tier)

    assert response.text == "A systems language."
    assert response.prompt_id == "ask.v1"
    [request] = llm.calls
    assert request.tier == tier
    assert request.prompt_id == "ask.v1"
    assert request.messages == [Message(role="user", content="What is Rust?")]
    assert request.max_tokens == 1024
    # The real ask.v1 file ships in the package and is what gets sent.
    assert request.system == load_prompt("ask", version=1).render()
    assert request.system.strip()


@pytest.mark.parametrize(("extra", "tier"), [([], ModelTier.CHEAP), (["--tier", "strong"], ModelTier.STRONG)])
def test_ask_command_prints_answer_and_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], extra: list[str], tier: ModelTier
) -> None:
    llm = FakeLLM(by_prompt={"What is Rust?": "A systems language."})
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["ask", "What is Rust?", *extra])

    out, err = capsys.readouterr()
    assert out == "A systems language.\n"
    assert err == "model=fake prompt=ask.v1 tokens=0 in / 0 out cost=$0.000000\n"
    assert llm.calls[0].tier == tier


def test_ask_command_rejects_unknown_tier(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def must_not_build_llm() -> LLMClient:
        pytest.fail("argparse should reject the tier before a client is built")

    monkeypatch.setattr(cli, "_make_llm", must_not_build_llm)
    with pytest.raises(SystemExit) as exc:
        cli.main(["ask", "What is Rust?", "--tier", "huge"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


class _FailingLLM:
    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMError("Anthropic rate limit reached, still failing after retries")


def test_ask_command_llm_error_becomes_clean_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def make_llm() -> LLMClient:
        return _FailingLLM()

    monkeypatch.setattr(cli, "_make_llm", make_llm)
    with pytest.raises(SystemExit) as exc:
        cli.main(["ask", "What is Rust?"])

    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "error: Anthropic rate limit reached, still failing after retries\n"


# evidence


class _StubInput(ToolInput):
    query: str


class _StubTool:
    name = "stub"
    description = "Returns scripted evidence"
    input_model = _StubInput

    def __init__(self, items: list[Evidence]) -> None:
        self.items = items
        self.inputs: list[_StubInput] = []

    async def run(self, input: _StubInput) -> list[Evidence]:
        self.inputs.append(input)
        return self.items


def _item(title: str, value: float | None = None) -> Evidence:
    return Evidence(
        source="stub",
        query="q",
        url=HttpUrl(f"https://example.com/{title}"),
        title=title,
        summary=f"{title} summary",
        value=value,
    )


def _use_registry(monkeypatch: pytest.MonkeyPatch, *tools: Tool[Any]) -> None:
    def make(settings: Settings, client: httpx.AsyncClient) -> ToolRegistry:
        return ToolRegistry(tools)

    monkeypatch.setattr(cli, "default_tools", make)


def test_stub_tool_satisfies_the_protocol() -> None:
    tool: Tool[_StubInput] = _StubTool([])  # pyright checks the assignment
    assert tool.name == "stub"


async def test_evidence_fills_the_query_field_and_returns_what_the_tool_found() -> None:
    items = [_item("a"), _item("b")]
    tool = _StubTool(items)

    result = await cli.evidence(ToolRegistry([tool]), "stub", "rust")

    assert result == items
    assert tool.inputs == [_StubInput(query="rust")]


async def test_evidence_unknown_tool_lists_the_available_ones() -> None:
    with pytest.raises(ToolError, match=r"unknown tool: nope \(available: stub\)"):
        await cli.evidence(ToolRegistry([_StubTool([])]), "nope", "rust")


def test_evidence_command_runs_a_real_tool_against_recorded_responses(
    respx_mock: respx.MockRouter, load_fixture: Callable[[str], Any], capsys: pytest.CaptureFixture[str]
) -> None:
    respx_mock.get(host="api.github.com", path="/search/repositories").respond(
        json=load_fixture("github_search/rust.json")
    )

    cli.main(["evidence", "github_search", "rust"])

    out, err = capsys.readouterr()
    blocks = out.rstrip("\n").split("\n\n")
    assert len(blocks) == 6
    assert blocks[0].startswith("[1] GitHub repositories matching 'rust'  (value: 683,239)\n")
    assert blocks[0].endswith("\n    https://github.com/search?q=rust&type=repositories")
    assert blocks[5].startswith("[6] rust-lang/rust  (value: 120,832)\n")
    assert err == "tool=github_search query='rust' evidence=6\n"


def test_evidence_command_with_no_results_prints_only_the_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_registry(monkeypatch, _StubTool([]))

    cli.main(["evidence", "stub", "rust"])

    assert capsys.readouterr() == ("", "tool=stub query='rust' evidence=0\n")


def test_evidence_command_unknown_tool_becomes_clean_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_registry(monkeypatch, _StubTool([]))
    with pytest.raises(SystemExit) as exc:
        cli.main(["evidence", "nope", "rust"])

    assert exc.value.code == 1
    assert capsys.readouterr() == ("", "error: unknown tool: nope (available: stub)\n")


def test_evidence_command_tool_failure_becomes_clean_message(
    respx_mock: respx.MockRouter, load_fixture: Callable[[str], Any], capsys: pytest.CaptureFixture[str]
) -> None:
    respx_mock.get(host="api.github.com", path="/repos/nope/missing").respond(
        404, json=load_fixture("github_repo/not_found.json")
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["evidence", "github_repo", "nope/missing"])

    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: github: unknown repository nope/missing")
    assert "Traceback" not in err
