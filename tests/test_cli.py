import argparse
import asyncio
import logging
import os
import re
import signal
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import uvicorn
from fastapi import FastAPI
from pydantic import HttpUrl
from uvicorn.importer import import_from_string

from techdebate import cli, session
from techdebate.config import Settings
from techdebate.errors import LLMError, ToolError
from techdebate.forum import Forum
from techdebate.llm import FakeLLM, LLMClient, LLMRequest, LLMResponse, ModelTier
from techdebate.llm import Message as LLMMessage
from techdebate.llm.fake import Predicate, Reply
from techdebate.logging_setup import setup_logging
from techdebate.models import (
    Evidence,
    LLMCall,
    Message,
    Notification,
    NotificationOutcome,
    NotificationReason,
    ReplyTo,
    ResearchPlan,
    SpeakDecision,
    User,
    UserKind,
)
from techdebate.orchestrator import Visit
from techdebate.prompts import load_prompt
from techdebate.roles import ROLE_SPECS
from techdebate.runner import agents_lock
from techdebate.spending import duration
from techdebate.store import open_store
from techdebate.tools import Tool, ToolInput, ToolRegistry
from techdebate.tree import Tree
from techdebate.web.server import WebServer


@pytest.mark.parametrize(
    ("argv", "verbose"),
    [
        (["run"], False),
        (["-v", "run"], True),
        (["run", "-v"], True),
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
        cli.main(["run"])

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
    assert request.messages == [LLMMessage(role="user", content="What is Rust?")]
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

    monkeypatch.setattr(cli, "default_tools", make)  # for evidence
    monkeypatch.setattr(session, "default_tools", make)  # for the commands that open a session


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


# run


_RESEARCH = ResearchPlan(
    speak=True,
    queries={"github_search": ["rust"], "github_repo": ["rust-lang/rust"], "pypi": ["rust"]},
    reason="In my area.",
)
_SKIP = ResearchPlan(speak=False, queries={"github_search": [], "github_repo": [], "pypi": []}, reason="Not my area.")


def _is_research(request: LLMRequest) -> bool:
    return request.prompt_id == "research.v1"


def _is_decide(request: LLMRequest) -> bool:
    return request.prompt_id == "decide.v1"


# The research and decide calls of an agent that comments on the post.
_COMMENTS: list[tuple[Predicate, Reply]] = [
    (_is_research, _RESEARCH),
    (_is_decide, SpeakDecision(reason="The data is new.", reply_to=[ReplyTo(place=0, point="It is used.")])),
]


def _is_write(request: LLMRequest) -> bool:
    return request.prompt_id == "developer.v1"


def _passes_at_research(display_name: str, tool_names: tuple[str, ...]) -> tuple[Predicate, Reply]:
    def matches(request: LLMRequest) -> bool:
        return _is_research(request) and request.system.startswith(f"You are the {display_name},")

    return matches, ResearchPlan(speak=False, queries={name: [] for name in tool_names}, reason="Not my area.")


# The Scientist and the Pollster look at every post too. In these tests they pass, so the rules after them script the
# Developer.
_OTHERS_PASS: list[tuple[Predicate, Reply]] = [
    _passes_at_research(role.display_name, role.tool_names) for role in ROLE_SPECS if role.name != "developer"
]


def _use_agent_tools(monkeypatch: pytest.MonkeyPatch, error: ToolError | None = None) -> None:
    """Stubs under the real tool names of every role, so default_roles keeps them all. With an error, every tool
    fails."""

    class _Failing(_StubTool):
        async def run(self, input: _StubInput) -> list[Evidence]:
            raise error or ToolError("unreachable")

    tools: list[Tool[Any]] = []
    for name in dict.fromkeys(name for role in ROLE_SPECS for name in role.tool_names):
        tool = _Failing([]) if error else _StubTool([_item(name)])
        tool.name = name
        tools.append(tool)
    _use_registry(monkeypatch, *tools)


def _use_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """A local .env can't change the result."""
    settings = Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]
    monkeypatch.setattr(cli, "get_settings", lambda: settings)


def test_ctrl_c_prints_one_line_and_exits_with_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(args: argparse.Namespace) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run", interrupt)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run"])

    assert exc.value.code == 130
    assert capsys.readouterr() == ("", "interrupted\n")


# The discussion in the database: post, comment, reply, show and notifications


def _use_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides: Any) -> Path:
    """Settings with a temporary database, and every role's tools so every role is in the registry."""
    db_path = tmp_path / "techdebate.db"
    _use_settings(monkeypatch, db_path=str(db_path), **overrides)
    _use_agent_tools(monkeypatch)
    return db_path


def _db_settings(db_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=str(db_path))  # pyright: ignore[reportCallIssue]


def _posts(db_path: Path) -> list[Message]:
    """The posts, newest first."""

    async def read() -> list[Message]:
        async with open_store(_db_settings(db_path)) as store:
            return await store.list_posts()

    return asyncio.run(read())


def _write_as(
    db_path: Path, username: str, text: str, reply_to: int | None = None, post: Message | None = None
) -> None:
    """Writes a comment as another user, through the forum, as an agent will: a top-level comment, or a reply to
    comment #reply_to, on `post` or on the newest post. The user is created as a real user if it does not exist yet."""

    async def write() -> None:
        async with open_store(_db_settings(db_path)) as store:
            user = await store.find_user(username)
            if user is None:
                user = User(username=username, kind=UserKind.REAL)
                await store.add_user(user)
            developer = await store.find_user("developer")
            assert developer is not None
            on = post or (await store.list_posts())[0]
            found = await store.get_post(on.id)
            assert found is not None
            parent = found.post if reply_to is None else found.comments[reply_to - 1]
            await Forum(store, [developer]).create_comment(user, parent.id, text)

    asyncio.run(write())


def _fails(capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    """The one-line error of a command that fails cleanly."""
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "Traceback" not in err
    return err


def test_post_creates_the_post_and_notifies_the_agents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)

    cli.main(["post", "  Is Rust growing?\n"])

    out, err = capsys.readouterr()

    async def saved() -> tuple[list[Message], list[str]]:
        async with open_store(_db_settings(db_path)) as store:
            developer = await store.find_user("developer")
            assert developer is not None
            return await store.list_posts(), [n.reason for n in await store.list_notifications(developer.id)]

    [post], reasons = asyncio.run(saved())
    assert (out, err) == ("", f"post={post.id[:8]}\n")
    assert post.text == "Is Rust growing?"
    assert reasons == [NotificationReason.NEW_POST]


def test_posts_lists_every_post_newest_first_with_its_short_id_comments_and_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["posts"])
    assert capsys.readouterr() == ("No posts.\n", "posts=0\n")

    cli.main(["post", "Is Rust growing?\nIt has more crates every year."])
    _write_as(db_path, "developer", "Yes.")
    cli.main(["post", "Is Go growing? " + "Go is used for cloud tools, and its module count keeps going up."])
    _write_as(db_path, "developer", "Yes.")
    _write_as(db_path, "developer", "More.")
    cli.main(["post", "Is Zig growing?"])
    capsys.readouterr()

    cli.main(["posts"])

    zig, go, rust = (post.id[:8] for post in _posts(db_path))
    assert capsys.readouterr() == (
        f"{zig}  0 comments  Is Zig growing?\n"
        f"{go}  2 comments  Is Go growing? Go is used for cloud tools, and its module…\n"
        f"{rust}  1 comment  Is Rust growing?\n",
        "posts=3\n",
    )


@pytest.mark.parametrize("argv", [["comment", "Yes."], ["reply", "1", "Yes."], ["show"], ["show", "--post", "ab"]])
def test_a_command_before_any_post_says_how_to_create_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: list[str]
) -> None:
    _use_db(monkeypatch, tmp_path)

    assert _fails(capsys, argv) == 'error: there is no post yet: create one with techdebate post "<text>"\n'


def test_notifications_before_any_post_lists_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path)

    cli.main(["notifications"])

    assert capsys.readouterr() == ("No notifications.\n", "notifications=0\n")


def test_show_prints_a_post_without_comments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path, username="alice")
    cli.main(["post", "Is Rust\ngrowing?"])
    capsys.readouterr()

    cli.main(["show"])

    [post] = _posts(tmp_path / "techdebate.db")
    assert capsys.readouterr() == ("alice:\nIs Rust\ngrowing?\n\n  No comments.\n", f"post={post.id[:8]} comments=0\n")


def test_comment_and_reply_print_the_new_number_and_show_prints_the_nesting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    _write_as(db_path, "developer", "Yes: [rust](https://github.com/rust-lang/rust).\n\nAnd growing.")  # 1
    capsys.readouterr()

    cli.main(["reply", "1", "Source?"])  # 2, a reply to the top-level comment
    _write_as(db_path, "developer", "GitHub search.", reply_to=2)  # 3, deeper
    cli.main(["reply", "3", "Thanks."])  # 4, deeper still
    cli.main(["comment", "A second thread."])  # 5
    short = _posts(db_path)[0].id[:8]
    assert capsys.readouterr().err == f"post={short} comment=#2\npost={short} comment=#4\npost={short} comment=#5\n"

    cli.main(["show"])

    out, err = capsys.readouterr()
    assert out == (
        "me:\n"
        "Is Rust growing?\n"
        "\n"
        "  #1 Developer [AI]:\n"
        "  Yes: [rust](https://github.com/rust-lang/rust).\n"
        "\n"
        "  And growing.\n"
        "\n"
        "    #2 me:\n"
        "    Source?\n"
        "\n"
        "    #3 Developer [AI], replying to #2:\n"
        "    GitHub search.\n"
        "\n"
        "    #4 me, replying to #3:\n"
        "    Thanks.\n"
        "\n"
        "  #5 me:\n"
        "  A second thread.\n"
    )
    assert err == f"post={short} comments=5\n"


def test_a_reply_to_an_unknown_number_is_refused_and_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    _write_as(db_path, "developer", "Yes.")
    capsys.readouterr()

    short = _posts(db_path)[0].id[:8]
    for number in ("2", "0", "-1"):
        assert _fails(capsys, ["reply", number, "Source?"]) == f"error: there is no comment #{number} on post {short}\n"
    cli.main(["show"])
    assert capsys.readouterr().err == f"post={short} comments=1\n"


def test_comment_reply_and_show_work_on_the_newest_post_or_the_one_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    cli.main(["post", "Is Go growing?"])
    go, rust = _posts(db_path)
    capsys.readouterr()

    cli.main(["comment", "On the newest."])
    cli.main(["comment", "--post", rust.id[:4].upper(), "On the older one."])  # any start of the ID, any case
    cli.main(["reply", "1", "--post", rust.id, "A reply there."])  # the full ID
    cli.main(["comment", "Newest again."])

    assert capsys.readouterr().err == (
        f"post={go.id[:8]} comment=#1\npost={rust.id[:8]} comment=#1\n"
        f"post={rust.id[:8]} comment=#2\npost={go.id[:8]} comment=#2\n"
    )
    cli.main(["show", "--post", rust.id[:6]])
    out, err = capsys.readouterr()
    assert out == "me:\nIs Rust growing?\n\n  #1 me:\n  On the older one.\n\n    #2 me:\n    A reply there.\n"
    assert err == f"post={rust.id[:8]} comments=2\n"
    cli.main(["show"])
    out, err = capsys.readouterr()
    assert out == "me:\nIs Go growing?\n\n  #1 me:\n  On the newest.\n\n  #2 me:\n  Newest again.\n"
    assert err == f"post={go.id[:8]} comments=2\n"


def _add_posts(db_path: Path, *ids: str) -> None:
    """Posts by me with the given IDs, saved directly so their IDs are known."""

    async def add() -> None:
        async with open_store(_db_settings(db_path)) as store:
            me = await store.find_user("me")
            assert me is not None
            for post_id in ids:
                await store.add_message(Message(id=post_id, author_id=me.id, text=f"Post {post_id}"))

    asyncio.run(add())


@pytest.mark.parametrize("argv", [["comment", "Yes."], ["reply", "1", "Yes."], ["show"], ["notifications"]])
def test_a_post_id_that_names_no_post_or_several_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: list[str]
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["posts"])  # creates the users
    first, second = "abcd1234" + "0" * 24, "abcd1234" + "1" * 24
    _add_posts(db_path, first, second)
    capsys.readouterr()

    assert _fails(capsys, [*argv, "--post", "ff"]) == "error: there is no post ff: techdebate posts lists them\n"
    assert _fails(capsys, [*argv, "--post", "abcd1234"]) == (
        f"error: --post abcd1234 matches several posts ({second}, {first}): give more of the ID\n"
    )
    # Long enough to name one. The reply finds that post, which has no comment to answer.
    if argv[0] == "reply":
        assert _fails(capsys, [*argv, "--post", "abcd12341"]) == "error: there is no comment #1 on post abcd1234\n"
    else:
        cli.main([*argv, "--post", "abcd12341"])
        assert capsys.readouterr().err.startswith(("post=abcd1234 ", "notifications=0"))


def test_a_reply_number_that_is_not_a_number_is_an_argument_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["reply", "one", "Source?"])

    assert exc.value.code == 2
    assert "invalid int value: 'one'" in capsys.readouterr().err


def test_a_blank_comment_is_refused_by_the_forum(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    capsys.readouterr()

    assert _fails(capsys, ["comment", "  \n"]) == "error: the text is blank\n"


def test_a_database_command_refuses_a_username_that_breaks_the_rule(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path, username="Andok")

    assert _fails(capsys, ["post", "Is Rust growing?"]).startswith("error: username 'Andok' is invalid")


def test_notifications_say_who_wrote_what_where_and_why_and_are_closed_as_seen(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    _write_as(db_path, "developer", "Yes.\nIt is.")  # 1: a top-level comment on my post
    cli.main(["reply", "1", "Source?"])  # 2
    _write_as(db_path, "developer", "GitHub.", reply_to=1)  # 3: a reply to #1, in a thread I commented in
    _write_as(db_path, "developer", "And PyPI.", reply_to=2)  # 4: a reply to my #2, deeper in thread #1
    _write_as(db_path, "ann", "What do you think, @Me?")  # 5: a top-level comment that mentions me
    _write_as(db_path, "ann", "@me again.", reply_to=4)  # 6: a reply that mentions me
    capsys.readouterr()

    cli.main(["notifications"])

    out, err = capsys.readouterr()
    short = _posts(db_path)[0].id[:8]
    expected = [
        (f"{short} #1 Developer [AI] commented on your post", "Yes.\nIt is."),
        (f"{short} #3 Developer [AI] replied to #1", "GitHub."),
        (f"{short} #4 Developer [AI] replied to #2, in thread #1", "And PyPI."),
        (f"{short} #5 ann mentioned you, on the post", "What do you think, @Me?"),
        (f"{short} #6 ann mentioned you, replying to #4", "@me again."),
    ]
    assert out == "\n\n".join(f"{header}:\n{text}" for header, text in expected) + "\n"
    assert err == "notifications=5\n"

    cli.main(["notifications"])
    assert capsys.readouterr() == ("No notifications.\n", "notifications=0\n")

    cli.main(["notifications", "--all"])
    out, err = capsys.readouterr()
    assert out == "\n\n".join(f"{header} (seen):\n{text}" for header, text in expected) + "\n"
    assert err == "notifications=5\n"


def test_notifications_with_all_closes_only_the_pending_ones(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    _write_as(db_path, "developer", "Yes.")
    cli.main(["notifications"])
    _write_as(db_path, "developer", "More.")
    capsys.readouterr()

    cli.main(["notifications", "--all"])

    short = _posts(db_path)[0].id[:8]
    assert capsys.readouterr().out == (
        f"{short} #1 Developer [AI] commented on your post (seen):\nYes.\n\n"
        f"{short} #2 Developer [AI] commented on your post:\nMore.\n"
    )
    cli.main(["notifications"])
    assert capsys.readouterr().out == "No notifications.\n"


def test_notifications_list_every_post_and_post_narrows_them_to_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    cli.main(["post", "Is Go growing?"])
    go, rust = _posts(db_path)
    _write_as(db_path, "developer", "Rust: yes.", post=rust)
    _write_as(db_path, "developer", "Go: yes.", post=go)
    _write_as(db_path, "developer", "Rust: more.", post=rust)
    capsys.readouterr()

    cli.main(["notifications", "--post", go.id[:5]])

    assert capsys.readouterr() == (
        f"{go.id[:8]} #1 Developer [AI] commented on your post:\nGo: yes.\n",
        "notifications=1\n",
    )
    cli.main(["notifications"])  # only the ones on the other post are left, oldest first
    assert capsys.readouterr() == (
        f"{rust.id[:8]} #1 Developer [AI] commented on your post:\nRust: yes.\n\n"
        f"{rust.id[:8]} #2 Developer [AI] commented on your post:\nRust: more.\n",
        "notifications=2\n",
    )


def test_a_mention_in_a_post_is_described_as_such() -> None:
    me = User(username="me", kind=UserKind.REAL)
    ann = User(username="ann", kind=UserKind.REAL)
    post = Message(author_id=ann.id, text="@me, is Rust growing?")
    notification = Notification(user_id=me.id, post_id=post.id, message_id=post.id, reason=NotificationReason.MENTION)

    header = cli._notification_header(notification, post, Tree(post, []), {ann.id: ann})  # pyright: ignore[reportPrivateUsage]

    assert header == f"{post.id[:8]} ann mentioned you in the post"


# run


def test_run_takes_no_argument() -> None:
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["run", "Rust"])  # pyright: ignore[reportPrivateUsage]


def test_run_before_any_post_is_quiet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path)
    llm = FakeLLM()
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run"])

    assert capsys.readouterr() == ("", "visits=0 comments=0 cost=$0.000000\n")
    assert llm.calls == []


def test_run_lets_the_agents_act_on_the_post_and_prints_each_visit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    text = "Developers use [it](https://example.com/github_search)."
    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, text)]))
    capsys.readouterr()

    cli.main(["run"])

    out, err = capsys.readouterr()
    short = _posts(tmp_path / "techdebate.db")[0].id[:8]
    assert sorted(out.splitlines()) == [
        f"{short} Developer [AI] commented: #1",
        f"{short} Pollster [AI] passed at research: Not my area.",
        f"{short} Scientist [AI] passed at research: Not my area.",
    ]
    assert err == "visits=3 comments=1 cost=$0.000000\n"
    cli.main(["show"])
    assert capsys.readouterr().out == f"me:\nIs Rust growing?\n\n  #1 Developer [AI]:\n  {text}\n"

    cli.main(["run"])  # the post is quiet now

    assert capsys.readouterr() == ("", "visits=0 comments=0 cost=$0.000000\n")


def test_run_prints_the_cost_of_every_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    class _Priced(FakeLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            response = await super().generate(request)
            return response.model_copy(update={"cost_usd": 0.25 if _is_write(request) else 0.01})

    _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    monkeypatch.setattr(cli, "_make_llm", lambda: _Priced(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, "Used.")]))
    capsys.readouterr()

    cli.main(["run"])

    # The Developer's research, decide and write, and the research calls of the two agents that pass.
    assert capsys.readouterr().err == "visits=3 comments=1 cost=$0.290000\n"


def test_run_still_prints_a_comment_when_every_tool_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path)
    _use_agent_tools(monkeypatch, ToolError("github is down"))
    cli.main(["post", "Is Rust growing?"])
    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, "Used.")]))
    capsys.readouterr()

    cli.main(["run"])

    out, err = capsys.readouterr()
    assert f"{_posts(tmp_path / 'techdebate.db')[0].id[:8]} Developer [AI] commented: #1" in out.splitlines()
    assert err.count("tool failed") == 3  # each failed tool still logs its warning
    assert err.splitlines()[-1] == "visits=3 comments=1 cost=$0.000000"


def test_run_records_a_failed_call_and_carries_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def make_llm() -> LLMClient:
        return _FailingLLM()

    _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    monkeypatch.setattr(cli, "_make_llm", make_llm)
    capsys.readouterr()

    cli.main(["run"])  # does not exit with an error

    out, err = capsys.readouterr()
    error = "Anthropic rate limit reached, still failing after retries"
    short = _posts(tmp_path / "techdebate.db")[0].id[:8]
    assert sorted(out.splitlines()) == [
        f"{short} {name} [AI] failed: {error}" for name in ("Developer", "Pollster", "Scientist")
    ]
    assert err.count("visit failed") == 3
    assert err.splitlines()[-1] == "visits=3 comments=0 cost=$0.000000"
    cli.main(["run"])  # the failed notifications are closed, so nothing is left
    assert capsys.readouterr().err == "visits=0 comments=0 cost=$0.000000\n"


def test_run_on_a_post_out_of_budget_makes_no_ai_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path, post_budget_usd=0.0)
    cli.main(["post", "Is Rust growing?"])
    llm = FakeLLM()
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)
    capsys.readouterr()

    cli.main(["run"])

    out, err = capsys.readouterr()
    line = "out of budget: the post has spent $0.0000 of its $0.00 budget"
    short = _posts(tmp_path / "techdebate.db")[0].id[:8]
    assert sorted(out.splitlines()) == [
        f"{short} {name} [AI] {line}" for name in ("Developer", "Pollster", "Scientist")
    ]
    assert err == "visits=3 comments=0 cost=$0.000000\n"
    assert llm.calls == []


def _spend(db_path: Path, username: str, post: Message, cost_usd: float, at: datetime | None = None) -> None:
    """An AI call the agent made on the post, stored as if made at `at` (now by default)."""

    async def add() -> None:
        async with open_store(_db_settings(db_path)) as store:
            agent = await store.find_user(username)
            assert agent is not None
            call = LLMCall(
                agent_id=agent.id,
                post_id=post.id,
                prompt_id="developer.v1",
                model="m",
                input_tokens=1,
                output_tokens=1,
                cost_usd=cost_usd,
                duration_s=0.1,
                created_at=at or datetime.now(UTC),
            )
            await store.add_llm_call(call)

    asyncio.run(add())


_RESETS = r"^Today \(UTC, resets in ([0-9]|1[0-9]|2[0-3])h [0-5]?[0-9]m\):$"


def test_budget_with_no_post_prints_today_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path)

    cli.main(["budget"])

    out, err = capsys.readouterr()
    first, *rows = out.splitlines()
    assert re.match(_RESETS, first)
    assert rows == [
        "  all agents      $0.0000 of $5.00   $5.0000 left",
        "  Developer [AI]  $0.0000 of $2.00   $2.0000 left",
        "  Scientist [AI]  $0.0000 of $2.00   $2.0000 left",
        "  Pollster [AI]   $0.0000 of $2.00   $2.0000 left",
    ]
    assert err == "spent_today=$0.000000\n"


def test_budget_prints_today_and_the_newest_post_or_another(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path, agent_budget_usd=0.30, post_budget_usd=0.50)
    cli.main(["post", "Is Rust growing?"])
    cli.main(["post", "Is Go growing?"])
    go, rust = _posts(db_path)
    _spend(db_path, "developer", rust, 0.20)
    _spend(db_path, "developer", go, 0.15)  # the developer's day is used up
    _spend(db_path, "scientist", rust, 0.3125)  # and the scientist's
    _spend(
        db_path, "pollster", rust, 1.25, at=datetime.now(UTC) - timedelta(days=1)
    )  # not today, though the Rust post counts it
    capsys.readouterr()

    cli.main(["budget"])

    out, err = capsys.readouterr()
    first, *rows = out.splitlines()
    assert re.match(_RESETS, first)
    assert rows == [
        "  all agents      $0.6625 of $5.00   $4.3375 left",
        "  Developer [AI]  $0.3500 of $0.30   used up",
        "  Scientist [AI]  $0.3125 of $0.30   used up",
        "  Pollster [AI]   $0.0000 of $0.30   $0.3000 left",
        "Post " + go.id[:8] + ":    $0.1500 of $0.50   $0.3500 left",
    ]
    assert err == f"spent_today=$0.662500 post={go.id[:8]}\n"

    cli.main(["budget", "--post", rust.id[:6].upper()])

    out, err = capsys.readouterr()
    assert out.splitlines()[-1] == "Post " + rust.id[:8] + ":    $1.7625 of $0.50   used up"
    assert err == f"spent_today=$0.662500 post={rust.id[:8]}\n"


def test_budget_on_an_unknown_post_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _use_db(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as raised:
        cli.main(["budget", "--post", "3f2a"])

    assert raised.value.code == 1
    assert capsys.readouterr().err == 'error: there is no post yet: create one with techdebate post "<text>"\n'


@pytest.mark.parametrize(
    ("left", "shown"),
    [
        (timedelta(hours=23, minutes=59, seconds=59), "23h 59m"),
        (timedelta(minutes=5), "0h 5m"),
        (timedelta(0), "0h 0m"),
    ],
)
def test_the_time_until_the_budgets_reset(left: timedelta, shown: str) -> None:
    assert duration(left) == shown


@pytest.mark.parametrize(
    ("outcome", "reason", "at_research", "comments", "line"),
    [
        (NotificationOutcome.COMMENTED, "New data.", False, (4, 5), "Developer [AI] commented: #4, #5"),
        (NotificationOutcome.PASSED, "Nothing new.", False, (), "Developer [AI] passed: Nothing new."),
        (NotificationOutcome.PASSED, "Not my area.", True, (), "Developer [AI] passed at research: Not my area."),
        (NotificationOutcome.LIMIT, "thread #1 is full", False, (), "Developer [AI] at its limit: thread #1 is full"),
        (NotificationOutcome.FAILED, "down", False, (), "Developer [AI] failed: down"),
        (NotificationOutcome.FAILED, "down", False, (3,), "Developer [AI] failed after writing #3: down"),
        (NotificationOutcome.BUDGET, "spent", False, (), "Developer [AI] out of budget: spent"),
        (NotificationOutcome.BUDGET, "spent", False, (3,), "Developer [AI] out of budget after writing #3: spent"),
    ],
)
def test_a_visit_line_names_the_agent_and_what_it_did(
    outcome: NotificationOutcome, reason: str, at_research: bool, comments: tuple[int, ...], line: str
) -> None:
    agent = User(username="developer", kind=UserKind.AGENT)
    visit = Visit(
        agent=agent,
        post_id="3f2a9c1e0b7d4e8a9c1f2b3d4e5f6a7b",
        outcome=outcome,
        reason=reason,
        notifications=1,
        comments=comments,
        at_research=at_research,
    )

    assert cli._visit_line(visit) == f"3f2a9c1e {line}"  # pyright: ignore[reportPrivateUsage]


def test_run_lines_name_the_post_of_each_visit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    cli.main(["post", "Is Go growing?"])
    go, rust = _posts(db_path)
    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, "Used.")]))
    capsys.readouterr()

    cli.main(["run"])

    out, err = capsys.readouterr()
    assert sorted(out.splitlines()) == sorted(
        f"{post.id[:8]} {line}"
        for post in (go, rust)
        for line in (
            "Developer [AI] commented: #1",
            "Pollster [AI] passed at research: Not my area.",
            "Scientist [AI] passed at research: Not my area.",
        )
    )
    assert err == "visits=6 comments=2 cost=$0.000000\n"


def test_run_goes_on_after_an_unexpected_error_and_prints_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    class _Buggy(FakeLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            if _is_write(request):
                raise KeyError("id")
            return await super().generate(request)

    _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    monkeypatch.setattr(cli, "_make_llm", lambda: _Buggy(rules=[*_OTHERS_PASS, *_COMMENTS]))
    capsys.readouterr()

    cli.main(["run"])  # does not exit with a traceback of its own

    out, err = capsys.readouterr()
    short = _posts(tmp_path / "techdebate.db")[0].id[:8]
    assert f"{short} Developer [AI] failed: unexpected error: KeyError('id')" in out.splitlines()
    assert "visit failed unexpectedly" in err and "Traceback" in err  # logged with its traceback
    assert err.splitlines()[-1] == "visits=3 comments=0 cost=$0.000000"


# agents


def _interrupt() -> None:
    os.kill(os.getpid(), signal.SIGINT)


def _interrupt_after(monkeypatch: pytest.MonkeyPatch, visits: int) -> None:
    """Sends Ctrl-C once `visits` visits have been printed."""
    printed: list[Visit] = []
    print_visit = cli._print_visit  # pyright: ignore[reportPrivateUsage]

    def counted(visit: Visit) -> None:
        print_visit(visit)
        printed.append(visit)
        if len(printed) == visits:
            _interrupt()

    monkeypatch.setattr(cli, "_print_visit", counted)


def test_agents_acts_on_what_is_pending_and_stops_cleanly_on_ctrl_c(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, "Used.")]))
    _interrupt_after(monkeypatch, 3)
    capsys.readouterr()

    cli.main(["agents"])  # returns: a clean stop exits with 0

    out, err = capsys.readouterr()
    short = _posts(db_path)[0].id[:8]
    assert sorted(out.splitlines()) == [
        f"{short} Developer [AI] commented: #1",
        f"{short} Pollster [AI] passed at research: Not my area.",
        f"{short} Scientist [AI] passed at research: Not my area.",
    ]
    assert err.splitlines() == [
        f"agents running on {db_path} (Ctrl-C to stop)",
        "stopping (Ctrl-C again to stop now)",
        "visits=3 comments=1 cost=$0.000000",
    ]


def test_the_first_ctrl_c_lets_the_visit_in_progress_finish(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    class _Interrupted(FakeLLM):
        """Ctrl-C comes while the Developer writes."""

        async def generate(self, request: LLMRequest) -> LLMResponse:
            if _is_write(request):
                _interrupt()
                await asyncio.sleep(0.1)  # the signal is handled while the visit is in progress
            return await super().generate(request)

    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    monkeypatch.setattr(cli, "_make_llm", lambda: _Interrupted(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, "Used.")]))
    capsys.readouterr()

    cli.main(["agents"])

    out, err = capsys.readouterr()
    assert f"{_posts(db_path)[0].id[:8]} Developer [AI] commented: #1" in out.splitlines()
    assert any(
        line.startswith("stopping: waiting for ") and line.endswith(" in progress (Ctrl-C again to stop now)")
        for line in err.splitlines()
    )
    assert err.splitlines()[-1].startswith("visits=")
    assert _pending_agent_notifications(db_path) == []


def test_a_second_ctrl_c_stops_at_once_and_leaves_the_visits_in_progress_pending(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    ended: list[Visit] = []
    print_visit = cli._print_visit  # pyright: ignore[reportPrivateUsage]

    def counted(visit: Visit) -> None:
        print_visit(visit)
        ended.append(visit)

    class _Interrupted(FakeLLM):
        """Ctrl-C twice while the Developer writes, a write that would take far too long. The Pollster's and the
        Scientist's visits may or may not have ended by then."""

        async def generate(self, request: LLMRequest) -> LLMResponse:
            if _is_write(request):
                _interrupt()
                await asyncio.sleep(0.05)
                _interrupt()
                await asyncio.sleep(30)
            return await super().generate(request)

    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    monkeypatch.setattr(cli, "_make_llm", lambda: _Interrupted(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, "Used.")]))
    monkeypatch.setattr(cli, "_print_visit", counted)
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        cli.main(["agents"])

    assert exc.value.code == 130
    assert capsys.readouterr().err.splitlines()[-1] == "interrupted"
    # Every visit cut short leaves its notification pending, to be handled next time: always the Developer's, and
    # the Pollster's or the Scientist's if it was still in progress.
    cut_short = {"developer", "pollster", "scientist"} - {visit.agent.username for visit in ended}
    assert "developer" in cut_short
    assert set(_pending_agent_notifications(db_path)) == cut_short


def _pending_agent_notifications(db_path: Path) -> list[str]:
    """The usernames of the agents with a pending notification."""

    async def read() -> list[str]:
        async with open_store(_db_settings(db_path)) as store:
            agents = await store.list_users(UserKind.AGENT)
            return [agent.username for agent in agents if await store.list_notifications(agent.id)]

    return asyncio.run(read())


@pytest.mark.parametrize("command", ["run", "agents"])
def test_the_agents_are_refused_while_another_process_runs_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, command: str
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    llm = FakeLLM()
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)
    capsys.readouterr()

    with agents_lock(_db_settings(db_path)):  # as another process would hold it
        err = _fails(capsys, [command])
        cli.main(["comment", "Still works."])  # the other commands do not take the lock

    assert err == (
        "error: the agents are already running on this database (techdebate agents, run or the web app): "
        "stop it first\n"
    )
    assert llm.calls == []


def test_agents_takes_no_argument() -> None:
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["agents", "Rust"])  # pyright: ignore[reportPrivateUsage]


# web


@dataclass
class _Served:
    """What `web` served: the app (or its import path, when it reloads), where, and whether it reloads."""

    app: object  # a FastAPI app, or its import path when it reloads
    host: str
    port: int
    reload: bool = False


def _serve(monkeypatch: pytest.MonkeyPatch, *, forced: bool = False) -> list[_Served]:
    """Replaces uvicorn's server, which would block, and its reloader. With `forced`, the server ends as after a second
    Ctrl-C. The agents get a fake model."""
    served: list[_Served] = []

    def serve(server: WebServer) -> None:
        served.append(_Served(server.config.app, server.config.host, server.config.port))
        server.forced = forced

    def run(app: str, **kwargs: Any) -> None:
        assert kwargs.pop("factory") is True
        assert kwargs.pop("reload") is True
        served.append(_Served(app, reload=True, **kwargs))

    monkeypatch.setattr(cli, "_serve", serve)
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM())
    return served


def _app(call: _Served) -> FastAPI:
    assert isinstance(call.app, FastAPI)
    return call.app


@pytest.mark.parametrize(
    ("argv", "port", "reload", "no_agents"),
    [(["web"], 8000, False, False), (["web", "--port", "9000", "--reload", "--no-agents"], 9000, True, True)],
)
def test_web_arguments(argv: list[str], port: int, reload: bool, no_agents: bool) -> None:
    args = cli._build_parser().parse_args(argv)  # pyright: ignore[reportPrivateUsage]

    assert (args.port, args.reload, args.no_agents) == (port, reload, no_agents)


def test_web_serves_the_app_with_the_agents_on_this_machine_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    served = _serve(monkeypatch)

    cli.main(["web", "--port", "9000"])

    [call] = served
    assert (call.host, call.port, call.reload) == ("127.0.0.1", 9000, False)
    assert _app(call).state.agents is True
    out, err = capsys.readouterr()
    assert out == ""
    assert err == f"web app on http://127.0.0.1:9000 with the agents, database {db_path} (Ctrl-C to stop)\n"


def test_web_without_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    served = _serve(monkeypatch)

    cli.main(["web", "--no-agents"])

    [call] = served
    assert _app(call).state.agents is False
    assert capsys.readouterr().err == (
        f"web app on http://127.0.0.1:8000 without the agents, database {db_path} (Ctrl-C to stop)\n"
    )


def test_web_opens_the_database_before_serving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    _serve(monkeypatch)

    cli.main(["web"])

    assert db_path.exists()


def test_web_while_the_agents_run_elsewhere_fails_in_one_line_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    served = _serve(monkeypatch)

    with agents_lock(_db_settings(db_path)):
        err = _fails(capsys, ["web"])

    assert err == (
        "error: the agents are already running on this database (techdebate agents, run or the web app): "
        "stop it first, or start the web app with --no-agents\n"
    )
    assert served == []


def test_web_without_agents_serves_while_the_agents_run_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    served = _serve(monkeypatch)

    with agents_lock(_db_settings(db_path)):
        cli.main(["web", "--no-agents"])

    assert len(served) == 1


@pytest.mark.parametrize(
    ("argv", "factory"), [(["web", "--reload"], cli.web_app), (["web", "--reload", "--no-agents"], cli.web_pages_app)]
)
def test_web_reload_hands_uvicorn_its_factory_by_its_import_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], factory: Callable[[], FastAPI]
) -> None:
    _use_db(monkeypatch, tmp_path)
    served = _serve(monkeypatch)

    cli.main(argv)

    [call] = served
    assert (call.host, call.port, call.reload) == ("127.0.0.1", 8000, True)
    assert isinstance(call.app, str)
    assert import_from_string(call.app) is factory


@pytest.mark.parametrize(("factory", "agents"), [(cli.web_app, True), (cli.web_pages_app, False)])
def test_the_reload_factories_build_the_app_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, factory: Callable[[], FastAPI], agents: bool
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    _serve(monkeypatch)

    app = factory()

    assert app.state.agents is agents
    assert app.state.settings.db_path == str(db_path)


def test_web_first_ctrl_c_stops_it_with_0_and_no_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_db(monkeypatch, tmp_path)
    _serve(monkeypatch)

    cli.main(["web"])  # returns, so the exit code is 0

    assert "interrupted" not in capsys.readouterr().err


def test_web_second_ctrl_c_prints_interrupted_and_exits_with_130(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_db(monkeypatch, tmp_path)
    _serve(monkeypatch, forced=True)

    with pytest.raises(SystemExit) as exc:
        cli.main(["web"])

    assert exc.value.code == 130
    assert capsys.readouterr().err.endswith("interrupted\n")


def test_web_with_a_bad_username_fails_in_one_line_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_db(monkeypatch, tmp_path, username="Andok")
    served = _serve(monkeypatch)

    assert _fails(capsys, ["web"]).startswith("error: username 'Andok' is invalid")
    assert served == []


# How the web server stops


@dataclass
class _Runner:
    in_progress: int
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


def _web_server(runner: _Runner | None) -> WebServer:
    app = FastAPI()
    app.state.runner = runner
    return WebServer(uvicorn.Config(app), app)


def test_the_web_servers_first_signal_lets_the_visits_finish(capsys: pytest.CaptureFixture[str]) -> None:
    runner = _Runner(in_progress=2)
    server = _web_server(runner)

    server.handle_exit(signal.SIGINT, None)

    assert server.should_exit
    assert not server.force_exit
    assert not server.forced
    assert not runner.cancelled
    assert capsys.readouterr().err == "stopping: waiting for 2 visits in progress (Ctrl-C again to stop now)\n"


@pytest.mark.parametrize("second", [signal.SIGINT, signal.SIGTERM])
def test_the_web_servers_second_signal_cancels_the_visits(second: signal.Signals) -> None:
    runner = _Runner(in_progress=1)
    server = _web_server(runner)

    server.handle_exit(signal.SIGTERM, None)
    server.handle_exit(second, None)

    assert server.forced
    assert server.force_exit
    assert runner.cancelled


def test_the_web_server_without_agents_says_nothing_on_a_signal(capsys: pytest.CaptureFixture[str]) -> None:
    server = _web_server(None)

    server.handle_exit(signal.SIGINT, None)
    server.handle_exit(signal.SIGINT, None)

    assert server.forced
    assert capsys.readouterr().err == ""


def test_the_web_server_does_not_send_the_signal_again_once_stopped() -> None:
    """uvicorn's would, which ends the process with SIGTERM's own code rather than 0."""
    server = _web_server(None)
    before = signal.getsignal(signal.SIGTERM)

    with server.capture_signals():
        assert signal.getsignal(signal.SIGTERM) == server.handle_exit
        server.handle_exit(signal.SIGTERM, None)

    assert signal.getsignal(signal.SIGTERM) == before  # restored, and the process is still here
