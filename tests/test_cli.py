import argparse
import logging

import pytest

from techdebate import cli
from techdebate.errors import LLMError, ToolError
from techdebate.llm import FakeLLM, LLMClient, LLMRequest, LLMResponse, Message, ModelTier
from techdebate.logging_setup import setup_logging
from techdebate.prompts import load_prompt


@pytest.mark.parametrize(
    ("argv", "verbose"),
    [
        (["run", "Rust"], False),
        (["-v", "run", "Rust"], True),
        (["run", "Rust", "-v"], True),
        (["ask", "What is Rust?", "-v"], True),
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
