"""Command-line entry point.

`techdebate run "<text>"` makes a post, lets the agents comment on it and prints both, `techdebate ask "<question>"`
checks the LLM setup, and `techdebate evidence <tool> "<query>"` runs one evidence tool.
"""

import argparse
import asyncio
import logging
import sys
import textwrap

from techdebate.config import get_settings
from techdebate.errors import TechDebateError
from techdebate.llm import AnthropicLLM, CostMeter, LLMClient, LLMRequest, LLMResponse, ModelTier
from techdebate.llm import Message as LLMMessage
from techdebate.logging_setup import setup_logging
from techdebate.models import Evidence, Message, User, UserKind
from techdebate.orchestrator import check_post
from techdebate.prompts import load_prompt
from techdebate.roles import default_roles
from techdebate.tools import ToolRegistry, default_tools, format_evidence, make_http_client
from techdebate.users import check_username, shown_name

log = logging.getLogger(__name__)

# Thinking counts against max_tokens, so this leaves room for both.
_ASK_MAX_TOKENS = 1024


async def ask(llm: LLMClient, question: str, tier: ModelTier) -> LLMResponse:
    prompt = load_prompt("ask", version=1)
    request = LLMRequest(
        tier=tier,
        system=prompt.render(),
        messages=[LLMMessage(role="user", content=question)],
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


async def _run_command(text: str, meter: CostMeter) -> tuple[Message, list[Message], dict[str, User]]:
    """The post, its comments, and every user by ID. The users exist only in memory for now."""
    settings = get_settings()
    me = User(username=check_username(settings.username), kind=UserKind.REAL)
    post = Message(author_id=me.id, text=text)
    async with make_http_client(settings) as client:
        tools = default_tools(settings, client)
        roles = default_roles(tools)
        agents = {role.name: User(username=role.name, kind=UserKind.AGENT) for role in roles.roles()}
        comments = await check_post(post, agents=agents, llm=meter, roles=roles, tools=tools, settings=settings)
    users = {user.id: user for user in (me, *agents.values())}
    return post, comments, users


def _format_post(post: Message, comments: list[Message], users: dict[str, User]) -> str:
    def block(message: Message) -> str:
        return f"{shown_name(users[message.author_id])}:\n{message.text}"

    below = "\n\n".join(block(comment) for comment in comments) if comments else "No comments."
    return f"{block(post)}\n\n{textwrap.indent(below, '  ')}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="techdebate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs and full tracebacks")
    # Same flag on subcommands; SUPPRESS keeps a `-v` given before the subcommand from being reset.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", parents=[common], help="Write a post and let every agent comment on it")
    run.add_argument("text", help="The text of the post")
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
        meter = CostMeter(_make_llm())
        post, comments, users = asyncio.run(_run_command(args.text, meter))
        print(_format_post(post, comments, users))
        print(f"comments={len(comments)} cost=${meter.total_usd:.6f}", file=sys.stderr)
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
