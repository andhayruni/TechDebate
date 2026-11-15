import argparse
import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import HttpUrl

from techdebate import cli
from techdebate.config import Settings
from techdebate.errors import LLMError, ToolError
from techdebate.forum import Forum
from techdebate.llm import FakeLLM, LLMClient, LLMRequest, LLMResponse, ModelTier
from techdebate.llm import Message as LLMMessage
from techdebate.llm.fake import Predicate, Reply
from techdebate.logging_setup import setup_logging
from techdebate.models import (
    Evidence,
    Message,
    Notification,
    NotificationReason,
    ReplyTo,
    ResearchPlan,
    SpeakDecision,
    User,
    UserKind,
)
from techdebate.prompts import load_prompt
from techdebate.roles import ROLE_SPECS
from techdebate.store import open_store
from techdebate.tools import Tool, ToolInput, ToolRegistry
from techdebate.tree import Tree


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


def test_run_has_no_delay_flag() -> None:
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["run", "Rust", "--no-delay"])  # pyright: ignore[reportPrivateUsage]


def test_run_command_prints_the_post_and_the_comment_with_its_ai_label(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_settings(monkeypatch)
    _use_agent_tools(monkeypatch)
    text = "Developers use [it](https://example.com/github_search)."
    llm = FakeLLM(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, text)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "Rust"])

    out, err = capsys.readouterr()
    assert out == f"me:\nRust\n\n  Developer [AI]:\n  {text}\n"
    assert err == "comments=1 cost=$0.000000\n"


def test_run_command_says_so_when_nobody_comments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_settings(monkeypatch, username="alice")
    _use_agent_tools(monkeypatch)
    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM(rules=[*_OTHERS_PASS, (_is_research, _SKIP)]))

    cli.main(["run", "Sourdough"])

    out, err = capsys.readouterr()
    assert out == "alice:\nSourdough\n\n  No comments.\n"
    assert err == "comments=0 cost=$0.000000\n"


def test_run_command_prints_the_cost_of_every_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Priced(FakeLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            response = await super().generate(request)
            return response.model_copy(update={"cost_usd": 0.25 if _is_write(request) else 0.01})

    _use_settings(monkeypatch)
    _use_agent_tools(monkeypatch)
    monkeypatch.setattr(cli, "_make_llm", lambda: _Priced(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, "Used.")]))

    cli.main(["run", "Rust"])

    # The Developer's research, decide and write, and the research calls of the two agents that pass.
    assert capsys.readouterr().err == "comments=1 cost=$0.290000\n"


def test_run_command_refuses_a_username_that_breaks_the_rule(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_settings(monkeypatch, username="Andok")
    _use_agent_tools(monkeypatch)
    llm = FakeLLM()
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "Rust"])

    assert exc.value.code == 1
    assert capsys.readouterr() == (
        "",
        "error: username 'Andok' is invalid: use 1 to 30 lowercase letters, digits or underscores\n",
    )
    assert llm.calls == []


def test_format_post_indents_each_comment_and_labels_only_agents() -> None:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    retired = User(username="retired", kind=UserKind.AGENT)
    post = Message(author_id=me.id, text="Is Rust\ngrowing?")
    comments = [
        Message(post_id=post.id, parent_id=post.id, author_id=author.id, text=text)
        for author, text in ((developer, "Yes.\nIt is."), (retired, "Two."))
    ]
    users = {user.id: user for user in (me, developer, retired)}

    assert cli._format_post(post, comments, users) == (  # pyright: ignore[reportPrivateUsage]
        "me:\nIs Rust\ngrowing?\n\n  Developer [AI]:\n  Yes.\n  It is.\n\n  retired [AI]:\n  Two."
    )


def test_run_command_still_prints_a_comment_when_every_tool_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_settings(monkeypatch)
    _use_agent_tools(monkeypatch, ToolError("github is down"))
    llm = FakeLLM(rules=[*_OTHERS_PASS, *_COMMENTS, (_is_write, "It is widely used.")])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "Rust"])

    out, err = capsys.readouterr()
    assert out == "me:\nRust\n\n  Developer [AI]:\n  It is widely used.\n"
    assert err.count("tool failed") == 3  # each failed tool still logs its warning
    assert err.splitlines()[-1] == "comments=1 cost=$0.000000"


def test_run_command_llm_error_becomes_clean_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def make_llm() -> LLMClient:
        return _FailingLLM()

    _use_settings(monkeypatch)
    _use_agent_tools(monkeypatch)
    monkeypatch.setattr(cli, "_make_llm", make_llm)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "Rust"])

    assert exc.value.code == 1
    assert capsys.readouterr() == ("", "error: Anthropic rate limit reached, still failing after retries\n")


def test_ctrl_c_prints_one_line_and_exits_with_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(args: argparse.Namespace) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run", interrupt)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "Rust"])

    assert exc.value.code == 130
    assert capsys.readouterr() == ("", "interrupted\n")


# The discussion in the database: post, comment, reply, show and notifications


def _use_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides: Any) -> Path:
    """Settings with a temporary database, and every role's tools so every role is in the registry."""
    db_path = tmp_path / "techdebate.db"
    _use_settings(monkeypatch, db_path=str(db_path), **overrides)
    _use_agent_tools(monkeypatch)
    return db_path


def _write_as(db_path: Path, username: str, text: str, reply_to: int | None = None) -> None:
    """Writes a comment as another user, through the forum, as an agent will: a top-level comment, or a reply to
    comment #reply_to. The user is created as a real user if it does not exist yet."""

    async def write() -> None:
        async with open_store(Settings(_env_file=None, db_path=str(db_path))) as store:  # pyright: ignore[reportCallIssue]
            user = await store.find_user(username)
            if user is None:
                user = User(username=username, kind=UserKind.REAL)
                await store.add_user(user)
            developer = await store.find_user("developer")
            assert developer is not None
            [post] = await store.list_posts()
            found = await store.get_post(post.id)
            assert found is not None
            parent = post if reply_to is None else found.comments[reply_to - 1]
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
        async with open_store(Settings(_env_file=None, db_path=str(db_path))) as store:  # pyright: ignore[reportCallIssue]
            developer = await store.find_user("developer")
            assert developer is not None
            return await store.list_posts(), [n.reason for n in await store.list_notifications(developer.id)]

    [post], reasons = asyncio.run(saved())
    assert (out, err) == ("", f"post={post.id}\n")
    assert post.text == "Is Rust growing?"
    assert reasons == [NotificationReason.NEW_POST]


def test_a_second_post_is_refused_and_names_the_database_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    capsys.readouterr()

    err = _fails(capsys, ["post", "Is Go growing?"])

    assert err == (
        "error: there is already a post, and only one is supported for now; "
        f"delete the database file {db_path} to start over\n"
    )
    cli.main(["show"])
    assert "Is Go growing?" not in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["comment", "Yes."], ["reply", "1", "Yes."], ["show"]])
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

    assert capsys.readouterr() == ("alice:\nIs Rust\ngrowing?\n\n  No comments.\n", "comments=0\n")


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
    assert capsys.readouterr().err == "comment=#2\ncomment=#4\ncomment=#5\n"

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
    assert err == "comments=5\n"


def test_a_reply_to_an_unknown_number_is_refused_and_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db_path = _use_db(monkeypatch, tmp_path)
    cli.main(["post", "Is Rust growing?"])
    _write_as(db_path, "developer", "Yes.")
    capsys.readouterr()

    for number in ("2", "0", "-1"):
        assert _fails(capsys, ["reply", number, "Source?"]) == f"error: there is no comment #{number} on the post\n"
    cli.main(["show"])
    assert capsys.readouterr().err == "comments=1\n"


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
    expected = [
        ("#1 Developer [AI] commented on your post", "Yes.\nIt is."),
        ("#3 Developer [AI] replied to #1", "GitHub."),
        ("#4 Developer [AI] replied to #2, in thread #1", "And PyPI."),
        ("#5 ann mentioned you, on the post", "What do you think, @Me?"),
        ("#6 ann mentioned you, replying to #4", "@me again."),
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

    assert capsys.readouterr().out == (
        "#1 Developer [AI] commented on your post (seen):\nYes.\n\n#2 Developer [AI] commented on your post:\nMore.\n"
    )
    cli.main(["notifications"])
    assert capsys.readouterr().out == "No notifications.\n"


def test_a_mention_in_a_post_is_described_as_such() -> None:
    me = User(username="me", kind=UserKind.REAL)
    ann = User(username="ann", kind=UserKind.REAL)
    post = Message(author_id=ann.id, text="@me, is Rust growing?")
    notification = Notification(user_id=me.id, post_id=post.id, message_id=post.id, reason=NotificationReason.MENTION)

    header = cli._notification_header(notification, post, Tree(post, []), {ann.id: ann})  # pyright: ignore[reportPrivateUsage]

    assert header == "ann mentioned you in the post"
