"""Command-line entry point.

`techdebate post`, `posts`, `comment`, `reply`, `show` and `notifications` take part in the discussions stored in the
database, as the user in `username`. A post is named by its short ID, the start of its ID, and the commands that work on
one post take `--post`. `techdebate run` lets the agents act on their notifications until none is pending, and
`techdebate agents` keeps them running until Ctrl-C. `techdebate ask "<question>"` checks the LLM setup, and
`techdebate evidence <tool> "<query>"` runs one evidence tool.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import textwrap
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from techdebate.config import Settings, get_settings
from techdebate.errors import CommandError, TechDebateError
from techdebate.forum import Forum
from techdebate.llm import AnthropicLLM, LLMClient, LLMRequest, LLMResponse, ModelTier
from techdebate.llm import Message as LLMMessage
from techdebate.logging_setup import setup_logging
from techdebate.models import Evidence, Message, Notification, NotificationOutcome, NotificationReason, User
from techdebate.orchestrator import AgentLoop, Visit
from techdebate.prompts import load_prompt
from techdebate.roles import RoleRegistry, default_roles
from techdebate.runner import AgentRunner, agents_lock
from techdebate.store import Store, open_store
from techdebate.tools import ToolRegistry, default_tools, format_evidence, make_http_client
from techdebate.tree import Tree
from techdebate.users import shown_name, sync_users
from techdebate.views import cut

log = logging.getLogger(__name__)

# Thinking counts against max_tokens, so this leaves room for both.
_ASK_MAX_TOKENS = 1024
# How much of a post's ID is shown. `--post` takes any unique prefix, so this is only what is printed.
_SHORT_ID_CHARS = 8
# How much of a post's first line `posts` shows.
_POST_START_CHARS = 60


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


# The discussion in the database


@dataclass(frozen=True)
class _Session:
    store: Store
    me: User  # the real user in `username`, who writes from the command line
    agents: dict[str, User]  # by role name, in registry order
    forum: Forum
    roles: RoleRegistry
    tools: ToolRegistry


@asynccontextmanager
async def _session(settings: Settings) -> AsyncGenerator[_Session]:
    """The store, with the users synced to the role registry, the forum that notifies its agents, and the tools. The
    HTTP client stays open for the whole command, so `run` can use the tools; the other commands make no request."""
    async with open_store(settings) as store, make_http_client(settings) as client:
        tools = default_tools(settings, client)
        roles = default_roles(tools)
        users = await sync_users(store, roles, settings.username)
        yield _Session(
            store=store,
            me=users.me,
            agents=users.agents,
            forum=Forum(store, users.agents.values()),
            roles=roles,
            tools=tools,
        )


def short_id(post_id: str) -> str:
    """How a post is named in what the commands print."""
    return post_id[:_SHORT_ID_CHARS]


async def _find_post(store: Store, ref: str | None) -> Message:
    """The post whose ID starts with `ref`, without regard to case, or the newest post without it."""
    posts = await store.list_posts()
    if not posts:
        raise CommandError('there is no post yet: create one with techdebate post "<text>"')
    if ref is None:
        return posts[0]
    prefix = ref.strip().lower()
    matches = [post for post in posts if prefix and post.id.startswith(prefix)]
    if not matches:
        raise CommandError(f"there is no post {ref}: techdebate posts lists them")
    if len(matches) > 1:
        # Full IDs, since the short ones of the matches may be the same.
        ids = ", ".join(post.id for post in matches)
        raise CommandError(f"--post {ref} matches several posts ({ids}): give more of the ID")
    return matches[0]


async def _tree(store: Store, post: Message) -> Tree:
    found = await store.get_post(post.id)
    assert found is not None, "the post was just read"
    return Tree.of(found)


async def _post_command(settings: Settings, text: str) -> Message:
    async with _session(settings) as session:
        return await session.forum.create_post(session.me, text)


async def _posts_command(settings: Settings) -> list[str]:
    """One line per post, newest first: its short ID, its number of comments and the start of its text."""
    async with _session(settings) as session:
        lines: list[str] = []
        for post in await session.store.list_posts():
            count = len((await _tree(session.store, post)).comments)
            comments = f"{count} comment{'' if count == 1 else 's'}"
            start = cut(post.text.splitlines()[0], _POST_START_CHARS)
            lines.append(f"{short_id(post.id)}  {comments}  {start}")
        return lines


async def _comment_command(
    settings: Settings, text: str, reply_to: int | None = None, post_ref: str | None = None
) -> tuple[Message, int]:
    """The post, and the number of the new comment on it: a top-level one, or a reply to comment #reply_to."""
    async with _session(settings) as session:
        post = await _find_post(session.store, post_ref)
        tree = await _tree(session.store, post)
        parent = tree.post if reply_to is None else tree.comment(reply_to)
        if parent is None:
            raise CommandError(f"there is no comment #{reply_to} on post {short_id(post.id)}")
        comment = await session.forum.create_comment(session.me, parent.id, text)
        return post, (await _tree(session.store, post)).number(comment)


async def _show_command(settings: Settings, post_ref: str | None = None) -> tuple[Message, str, int]:
    """The post, the printed discussion, and its number of comments."""
    async with _session(settings) as session:
        post = await _find_post(session.store, post_ref)
        tree = await _tree(session.store, post)
        users = {user.id: user for user in await session.store.list_users()}
        return post, _format_discussion(tree, users), len(tree.comments)


def _agent_loop(session: _Session, settings: Settings) -> AgentLoop:
    return AgentLoop(
        session.store,
        session.forum,
        session.agents,
        llm=_make_llm(),
        roles=session.roles,
        tools=session.tools,
        settings=settings,
    )


def _print_visit(visit: Visit) -> None:
    print(_visit_line(visit), flush=True)


async def _run_command(settings: Settings) -> list[Visit]:
    """Lets the agents act until no notification is pending, printing one line per visit as it ends."""
    async with _session(settings) as session:
        with agents_lock(settings):
            return await _agent_loop(session, settings).run(on_visit=_print_visit)


def _full_path(path: str) -> str:
    return os.path.abspath(path)


# How the agents command is stopped: the first signal lets the visits in progress finish, and a second ends them.
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


async def _agents_command(settings: Settings) -> tuple[list[Visit], bool]:
    """Keeps the agents running, printing one line per visit as it ends, until Ctrl-C or SIGTERM. Returns the visits,
    and whether a second signal cut the visits in progress short."""
    async with _session(settings) as session:
        with agents_lock(settings):
            runner = AgentRunner(_agent_loop(session, settings), poll_s=settings.agent_poll_s, on_visit=_print_visit)
            session.forum.subscribe(runner.wake)
            stop = asyncio.Event()
            forced = False

            def on_signal() -> None:
                nonlocal forced
                if not stop.is_set():
                    busy = runner.in_progress
                    waiting = f": waiting for {busy} visit{'' if busy == 1 else 's'} in progress" if busy else ""
                    print(f"stopping{waiting} (Ctrl-C again to stop now)", file=sys.stderr, flush=True)
                    stop.set()
                else:
                    forced = True
                    runner.cancel()

            loop = asyncio.get_running_loop()
            for sig in _STOP_SIGNALS:
                loop.add_signal_handler(sig, on_signal)
            try:
                print(f"agents running on {_full_path(settings.db_path)} (Ctrl-C to stop)", file=sys.stderr, flush=True)
                await runner.start()
                await stop.wait()
                await runner.stop()
            finally:
                for sig in _STOP_SIGNALS:
                    loop.remove_signal_handler(sig)
            return runner.visits, forced


def _visit_line(visit: Visit) -> str:
    """The post, who made the visit, and what it did, in one line."""
    name = f"{short_id(visit.post_id)} {shown_name(visit.agent)}"
    numbers = ", ".join(f"#{n}" for n in visit.comments)
    if visit.outcome is NotificationOutcome.COMMENTED:
        return f"{name} commented: {numbers}"
    if visit.outcome is NotificationOutcome.PASSED:
        return f"{name} passed{' at research' if visit.at_research else ''}: {visit.reason}"
    if visit.outcome is NotificationOutcome.LIMIT:
        return f"{name} at its limit: {visit.reason}"
    after = f" after writing {numbers}" if visit.comments else ""
    if visit.outcome is NotificationOutcome.BUDGET:
        return f"{name} out of budget{after}: {visit.reason}"
    return f"{name} failed{after}: {visit.reason}"


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


async def _notifications_command(settings: Settings, *, include_closed: bool, post_ref: str | None = None) -> int:
    """Prints my notifications on every post, or on one with `post_ref`, oldest first, then closes the pending ones it
    printed as seen. Returns how many it printed."""
    async with _session(settings) as session:
        store = session.store
        notifications = await store.list_notifications(session.me.id, include_closed=include_closed)
        if post_ref is not None:
            post = await _find_post(store, post_ref)
            notifications = [notification for notification in notifications if notification.post_id == post.id]
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
    """The post, who wrote the message, where, and why it is listed, in plain words."""
    seen = "" if notification.pending else " (seen)"
    post = short_id(tree.post.id)
    author = shown_name(users[message.author_id])
    if message.parent_id is None:  # the post itself
        why = "mentioned you in the post" if notification.reason is NotificationReason.MENTION else "posted"
        return f"{post} {author} {why}{seen}"
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
    return f"{post} #{tree.number(message)} {author} {why}{seen}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="techdebate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs and full tracebacks")
    # Same flag on subcommands; SUPPRESS keeps a `-v` given before the subcommand from being reset.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", parents=[common], help="Let the agents act on their notifications until none is pending")
    sub.add_parser(
        "agents", parents=[common], help="Keep the agents running, acting on notifications as they arrive, until Ctrl-C"
    )
    ask_cmd = sub.add_parser("ask", parents=[common], help="Ask the model one question (checks the API key and client)")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--tier", choices=[t.value for t in ModelTier], default=ModelTier.CHEAP.value)
    evidence_cmd = sub.add_parser("evidence", parents=[common], help="Run one evidence tool and print what it finds")
    # A free string, not `choices`: the registry knows which tools are available and lists them on an unknown name.
    evidence_cmd.add_argument("tool", help="Tool name, e.g. github_search or github_repo")
    evidence_cmd.add_argument("query", help="Fills the tool's query field")

    # The post a command works on, by its ID or any unique start of it, as `posts` prints it.
    on_post = argparse.ArgumentParser(add_help=False)
    on_post.add_argument("--post", metavar="ID", help="The post, by its ID or the start of it (default: the newest)")

    post = sub.add_parser("post", parents=[common], help="Create a post")
    post.add_argument("text", help="The text of the post")
    sub.add_parser("posts", parents=[common], help="List the posts, newest first, with their short IDs")
    comment = sub.add_parser("comment", parents=[common, on_post], help="Add a top-level comment to a post")
    comment.add_argument("text", help="The text of the comment")
    reply = sub.add_parser("reply", parents=[common, on_post], help="Reply to comment #n on a post")
    reply.add_argument("number", type=int, metavar="n", help="The number of the comment to answer, as show prints it")
    reply.add_argument("text", help="The text of the reply")
    sub.add_parser("show", parents=[common, on_post], help="Print a post and its threads")
    notifications = sub.add_parser(
        "notifications", parents=[common], help="List my pending notifications and mark them as seen"
    )
    notifications.add_argument("--all", action="store_true", help="List the ones already seen too")
    notifications.add_argument(
        "--post", metavar="ID", help="Only the ones on this post, by its ID or the start of it (default: every post)"
    )
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.command == "run":
        _print_summary(asyncio.run(_run_command(get_settings())))
    if args.command == "agents":
        visits, forced = asyncio.run(_agents_command(get_settings()))
        if forced:
            raise KeyboardInterrupt
        _print_summary(visits)
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
        print(f"post={short_id(post.id)}", file=sys.stderr)
    if args.command == "posts":
        lines = asyncio.run(_posts_command(get_settings()))
        print("\n".join(lines) if lines else "No posts.")
        print(f"posts={len(lines)}", file=sys.stderr)
    if args.command in ("comment", "reply"):
        reply_to = args.number if args.command == "reply" else None
        post, number = asyncio.run(_comment_command(get_settings(), args.text, reply_to, args.post))
        print(f"post={short_id(post.id)} comment=#{number}", file=sys.stderr)
    if args.command == "show":
        post, text, count = asyncio.run(_show_command(get_settings(), args.post))
        print(text)
        print(f"post={short_id(post.id)} comments={count}", file=sys.stderr)
    if args.command == "notifications":
        count = asyncio.run(_notifications_command(get_settings(), include_closed=args.all, post_ref=args.post))
        print(f"notifications={count}", file=sys.stderr)


def _print_summary(visits: list[Visit]) -> None:
    comments = sum(len(visit.comments) for visit in visits)
    cost = sum(visit.cost_usd for visit in visits)
    print(f"visits={len(visits)} comments={comments} cost=${cost:.6f}", file=sys.stderr)


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
