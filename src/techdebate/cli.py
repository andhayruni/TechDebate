"""Command-line entry point. `techdebate ask "<question>"` checks the LLM setup; `run` arrives with the thin slice."""

import argparse
import asyncio
import logging
import sys

from techdebate.config import get_settings
from techdebate.errors import TechDebateError
from techdebate.llm import AnthropicLLM, LLMClient, LLMRequest, LLMResponse, Message, ModelTier
from techdebate.logging_setup import setup_logging
from techdebate.prompts import load_prompt

log = logging.getLogger(__name__)

# Thinking counts against max_tokens, so this leaves room for both.
_ASK_MAX_TOKENS = 1024


async def ask(llm: LLMClient, question: str, tier: ModelTier) -> LLMResponse:
    prompt = load_prompt("ask", version=1)
    request = LLMRequest(
        tier=tier,
        system=prompt.render(),
        messages=[Message(role="user", content=question)],
        max_tokens=_ASK_MAX_TOKENS,
        prompt_id=prompt.id,
    )
    return await llm.generate(request)


def _make_llm() -> LLMClient:
    return AnthropicLLM(get_settings())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="techdebate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs and full tracebacks")
    # Same flag on subcommands; SUPPRESS keeps a `-v` given before the subcommand from being reset.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", parents=[common], help="Run a debate on a technology")
    run.add_argument("topic")
    ask_cmd = sub.add_parser("ask", parents=[common], help="Ask the model one question (checks the API key and client)")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--tier", choices=[t.value for t in ModelTier], default=ModelTier.CHEAP.value)
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.command == "run":
        raise SystemExit("Debate loop not implemented yet.")
    if args.command == "ask":
        response = asyncio.run(ask(_make_llm(), args.question, ModelTier(args.tier)))
        print(response.text)
        usage = response.usage
        print(
            f"model={response.model} prompt={response.prompt_id} "
            f"tokens={usage.input_tokens} in / {usage.output_tokens} out cost=${response.cost_usd:.6f}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        _run(args)
    except TechDebateError as e:
        log.debug("command failed command=%s", args.command, exc_info=True)
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
