"""Command-line entry point.

`techdebate run "<topic>"` runs a debate and prints it, `techdebate ask "<question>"` checks the LLM setup, and
`techdebate evidence <tool> "<query>"` runs one evidence tool.
"""

import argparse
import asyncio
import logging
import sys

from techdebate.config import get_settings
from techdebate.errors import TechDebateError
from techdebate.llm import AnthropicLLM, LLMClient, LLMRequest, LLMResponse, Message, ModelTier
from techdebate.logging_setup import setup_logging
from techdebate.models import Debate, Evidence
from techdebate.orchestrator import run_debate
from techdebate.prompts import load_prompt
from techdebate.roles import default_roles, display_name
from techdebate.tools import ToolRegistry, default_tools, format_evidence, make_http_client

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


async def evidence(registry: ToolRegistry, tool_name: str, query: str) -> list[Evidence]:
    tool = registry.get(tool_name)
    return await tool.run(tool.input_model(query=query))


async def _evidence_command(tool_name: str, query: str) -> list[Evidence]:
    settings = get_settings()
    async with make_http_client(settings) as client:
        return await evidence(default_tools(settings, client), tool_name, query)


async def _debate_command(topic: str, no_delay: bool) -> Debate:
    settings = get_settings()
    if no_delay:
        settings = settings.model_copy(update={"read_delays": False})  # a copy, since get_settings is cached
    async with make_http_client(settings) as client:
        tools = default_tools(settings, client)
        return await run_debate(topic, llm=_make_llm(), roles=default_roles(tools), tools=tools, settings=settings)


def _format_debate(debate: Debate) -> str:
    if not debate.turns:
        return "Nobody joined the discussion."
    return "\n\n".join(f"{display_name(turn.analyst)}:\n{turn.text}" for turn in debate.turns)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="techdebate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs and full tracebacks")
    # Same flag on subcommands; SUPPRESS keeps a `-v` given before the subcommand from being reset.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", parents=[common], help="Run a debate on a technology")
    run.add_argument("topic")
    run.add_argument("--no-delay", action="store_true", help="Skip the roles' reading delays")
    ask_cmd = sub.add_parser("ask", parents=[common], help="Ask the model one question (checks the API key and client)")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--tier", choices=[t.value for t in ModelTier], default=ModelTier.CHEAP.value)
    evidence_cmd = sub.add_parser("evidence", parents=[common], help="Run one evidence tool and print what it finds")
    # A free string, not `choices`: the registry knows which tools are available and lists them on an unknown name.
    evidence_cmd.add_argument("tool", help="Tool name, e.g. github_search or github_repo")
    evidence_cmd.add_argument("query", help="Fills the tool's query field")
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.command == "run":
        debate = asyncio.run(_debate_command(args.topic, args.no_delay))
        print(_format_debate(debate))
        print(f"topic={debate.topic!r} turns={len(debate.turns)} cost=${debate.total_cost_usd:.6f}", file=sys.stderr)
    if args.command == "ask":
        response = asyncio.run(ask(_make_llm(), args.question, ModelTier(args.tier)))
        print(response.text)
        usage = response.usage
        print(
            f"model={response.model} prompt={response.prompt_id} "
            f"tokens={usage.input_tokens} in / {usage.output_tokens} out cost=${response.cost_usd:.6f}",
            file=sys.stderr,
        )
    if args.command == "evidence":
        items = asyncio.run(_evidence_command(args.tool, args.query))
        if items:
            print(format_evidence(items))
        print(f"tool={args.tool} query={args.query!r} evidence={len(items)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        _run(args)
    except TechDebateError as e:
        log.debug("command failed command=%s", args.command, exc_info=True)
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None
