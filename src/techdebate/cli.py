"""Command-line entry point.

`techdebate post`, `comment`, `reply`, `show` and `notifications` take part in the discussion stored in the database, as
the user in `username`. `techdebate run "<text>"` makes a post in memory, lets the agents comment on it and prints
both, `techdebate ask "<question>"` checks the LLM setup, and `techdebate evidence <tool> "<query>"` runs one evidence
tool.
"""

import argparse
import asyncio
import logging
import sys
import textwrap
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from techdebate.config import Settings, get_settings
from techdebate.errors import CommandError, TechDebateError
from techdebate.forum import Forum
from techdebate.llm import AnthropicLLM, CostMeter, LLMClient, LLMRequest, LLMResponse, ModelTier
from techdebate.llm import Message as LLMMessage
from techdebate.logging_setup import setup_logging
from techdebate.models import Evidence, Message, Notification, NotificationOutcome, NotificationReason, User, UserKind
from techdebate.orchestrator import check_post
from techdebate.prompts import load_prompt
from techdebate.roles import default_roles
from techdebate.store import Store, open_store
from techdebate.tools import ToolRegistry, default_tools, format_evidence, make_http_client
from techdebate.tree import Tree
from techdebate.users import check_username, shown_name, sync_users

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
        comments = await check_post(
            post, author=me, agents=agents, llm=meter, roles=roles, tools=tools, settings=settings
        )
    users = {user.id: user for user in (me, *agents.values())}
    return post, comments, users


def _format_post(post: Message, comments: list[Message], users: dict[str, User]) -> str:
    def block(message: Message) -> str:
        return f"{shown_name(users[message.author_id])}:\n{message.text}"

    below = "\n\n".join(block(comment) for comment in comments) if comments else "No comments."
    return f"{block(post)}\n\n{textwrap.indent(below, '  ')}"


# The discussion in the database


@dataclass(frozen=True)
class _Session:
    store: Store
    me: User  # the real user in `username`, who writes from the command line
    forum: Forum


@asynccontextmanager
async def _session(settings: Settings) -> AsyncGenerator[_Session]:
    """The store, with the users synced to the role registry that `run` builds, and the forum that notifies its
    agents."""
    async with open_store(settings) as store:
        # The tools are only built to learn which roles are available: no request is made.
        async with make_http_client(settings) as client:
            roles = default_roles(default_tools(settings, client))
        users = await sync_users(store, roles, settings.username)
        yield _Session(store=store, me=users.me, forum=Forum(store, users.agents.values()))


async def _the_post(store: Store) -> Tree:
    """The post with its comments. There is a single post until several posts are supported."""
    posts = await store.list_posts()
    found = await store.get_post(posts[0].id) if posts else None
    if found is None:
        raise CommandError('there is no post yet: create one with techdebate post "<text>"')
    return Tree.of(found)


async def _number(store: Store, comment: Message) -> int:
    return (await _the_post(store)).number(comment)


async def _post_command(settings: Settings, text: str) -> Message:
    async with _session(settings) as session:
        if await session.store.list_posts():
            raise CommandError(
                "there is already a post, and only one is supported for now; "
                f"delete the database file {settings.db_path} to start over"
            )
        return await session.forum.create_post(session.me, text)


async def _comment_command(settings: Settings, text: str, reply_to: int | None = None) -> int:
    """The number of the new comment: a top-level one, or a reply to comment #reply_to."""
    async with _session(settings) as session:
        tree = await _the_post(session.store)
        parent = tree.post if reply_to is None else tree.comment(reply_to)
        if parent is None:
            raise CommandError(f"there is no comment #{reply_to} on the post")
        comment = await session.forum.create_comment(session.me, parent.id, text)
        return await _number(session.store, comment)


async def _show_command(settings: Settings) -> tuple[str, int]:
    """The printed discussion, and its number of comments."""
    async with _session(settings) as session:
        tree = await _the_post(session.store)
        users = {user.id: user for user in await session.store.list_users()}
        return _format_discussion(tree, users), len(tree.comments)


def _format_discussion(tree: Tree, users: dict[str, User]) -> str:
    """The post, then each thread: its top-level comment one level in, and every reply under it a second level in,
    naming the comment it answers when that is not the top-level one."""

    def block(header: str, text: str, level: int) -> str:
        return textwrap.indent(f"{header}\n{text}", "  " * level)

    blocks = [block(f"{shown_name(users[tree.post.author_id])}:", tree.post.text, 0)]
    if not tree.comments:
        blocks.append(textwrap.indent("No comments.", "  "))
    for top, *replies in tree.threads():
        blocks.append(block(f"{_comment_header(tree, top, users)}:", top.text, 1))
        for reply in replies:
            parent = tree.parent(reply)
            answers = "" if parent.id == top.id else f", replying to #{tree.number(parent)}"
            blocks.append(block(f"{_comment_header(tree, reply, users)}{answers}:", reply.text, 2))
    return "\n\n".join(blocks)


def _comment_header(tree: Tree, comment: Message, users: dict[str, User]) -> str:
    return f"#{tree.number(comment)} {shown_name(users[comment.author_id])}"


async def _notifications_command(settings: Settings, *, include_closed: bool) -> int:
    """Prints my notifications, oldest first, then closes the pending ones it printed as seen. Returns how many it
    printed."""
    async with _session(settings) as session:
        store = session.store
        notifications = await store.list_notifications(session.me.id, include_closed=include_closed)
        users = {user.id: user for user in await store.list_users()}
        trees: dict[str, Tree] = {}
        blocks: list[str] = []
        for notification in notifications:
            if notification.post_id not in trees:
                found = await store.get_post(notification.post_id)
                assert found is not None, "a notification's post is saved before it"
                trees[notification.post_id] = Tree.of(found)
            tree = trees[notification.post_id]
            message = await store.get_message(notification.message_id)
            assert message is not None, "a notification's message is saved with it"
            blocks.append(f"{_notification_header(notification, message, tree, users)}:\n{message.text}")
        print("\n\n".join(blocks) if blocks else "No notifications.")
        pending = [notification.id for notification in notifications if notification.pending]
        await store.close_notifications(pending, NotificationOutcome.SEEN)
        return len(notifications)


def _notification_header(notification: Notification, message: Message, tree: Tree, users: dict[str, User]) -> str:
    """Who wrote the message, where, and why it is listed, in plain words."""
    seen = "" if notification.pending else " (seen)"
    author = shown_name(users[message.author_id])
    if message.parent_id is None:  # the post itself
        why = "mentioned you in the post" if notification.reason is NotificationReason.MENTION else "posted"
        return f"{author} {why}{seen}"
    parent = tree.parent(message)
    if notification.reason is NotificationReason.THREAD:
        top = tree.top_level(message)
        where = "" if parent.id == top.id else f", in thread #{tree.number(top)}"
        why = f"replied to #{tree.number(parent)}{where}"
    elif notification.reason is NotificationReason.MENTION:
        where = "on the post" if parent.id == tree.post.id else f"replying to #{tree.number(parent)}"
        why = f"mentioned you, {where}"
    else:  # a top-level comment, which a real user hears of on its own post
        why = "commented on your post"
    return f"#{tree.number(message)} {author} {why}{seen}"


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

    post = sub.add_parser("post", parents=[common], help="Create the post (one for now)")
    post.add_argument("text", help="The text of the post")
    comment = sub.add_parser("comment", parents=[common], help="Add a top-level comment to the post")
    comment.add_argument("text", help="The text of the comment")
    reply = sub.add_parser("reply", parents=[common], help="Reply to comment #n")
    reply.add_argument("number", type=int, metavar="n", help="The number of the comment to answer, as show prints it")
    reply.add_argument("text", help="The text of the reply")
    sub.add_parser("show", parents=[common], help="Print the post and its threads")
    notifications = sub.add_parser(
        "notifications", parents=[common], help="List my pending notifications and mark them as seen"
    )
    notifications.add_argument("--all", action="store_true", help="List the ones already seen too")
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
    if args.command == "post":
        post = asyncio.run(_post_command(get_settings(), args.text))
        print(f"post={post.id}", file=sys.stderr)
    if args.command in ("comment", "reply"):
        reply_to = args.number if args.command == "reply" else None
        number = asyncio.run(_comment_command(get_settings(), args.text, reply_to))
        print(f"comment=#{number}", file=sys.stderr)
    if args.command == "show":
        text, count = asyncio.run(_show_command(get_settings()))
        print(text)
        print(f"comments={count}", file=sys.stderr)
    if args.command == "notifications":
        count = asyncio.run(_notifications_command(get_settings(), include_closed=args.all))
        print(f"notifications={count}", file=sys.stderr)


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
