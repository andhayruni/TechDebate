import asyncio
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import HttpUrl

from techdebate.config import Settings
from techdebate.llm import FakeLLM
from techdebate.models import (
    AgentVisit,
    Evidence,
    LLMCall,
    Message,
    Notification,
    NotificationOutcome,
    NotificationReason,
    ReplyTo,
    Research,
    ResearchPlan,
    SpeakDecision,
    ToolCall,
    ToolCallOutcome,
    ToolResult,
    User,
    UserKind,
)
from techdebate.orchestrator import AgentLoop
from techdebate.roles import RoleRegistry, RoleSpec
from techdebate.session import open_session
from techdebate.spending import spent_on_post
from techdebate.store import PostWithComments, open_store
from techdebate.tools import ToolInput, ToolRegistry
from techdebate.tree import Tree
from techdebate.web import create_app
from techdebate.web.trace import DecideAnswer, ResearchAnswer, TracePage, WriteAnswer, trace_page

BASE_URL = "http://127.0.0.1"
_START = datetime(2026, 11, 21, 14, 5, 32, tzinfo=UTC)

# The records of a small discussion, built by hand: the post, the Developer's #1, my reply #2 under it, and the
# Scientist's #3 on the post.
_ME = User(username="me", kind=UserKind.REAL)
_DEV = User(username="developer", kind=UserKind.AGENT)
_SCI = User(username="scientist", kind=UserKind.AGENT)
_USERS = {user.id: user for user in (_ME, _DEV, _SCI)}
_POST = Message(author_id=_ME.id, text="Is **Rust** growing?")
_C1 = Message(post_id=_POST.id, parent_id=_POST.id, author_id=_DEV.id, text="Yes, downloads doubled.")
_C2 = Message(post_id=_POST.id, parent_id=_C1.id, author_id=_ME.id, text="@scientist any papers?")
_C3 = Message(post_id=_POST.id, parent_id=_POST.id, author_id=_SCI.id, text="Papers grow too.")
_TREE = Tree.of(PostWithComments(post=_POST, comments=[_C1, _C2, _C3]))


def _visit(
    agent: User = _DEV, outcome: NotificationOutcome | None = NotificationOutcome.PASSED, **fields: Any
) -> AgentVisit:
    ended = None if outcome is None else _START + timedelta(seconds=12.3)
    return AgentVisit(agent_id=agent.id, post_id=_POST.id, started_at=_START, ended_at=ended, outcome=outcome, **fields)


def _call(visit: AgentVisit | None, prompt_id: str, output: str | None, **fields: object) -> LLMCall:
    values: dict[str, object] = {
        "agent_id": _DEV.id,
        "post_id": _POST.id,
        "visit_id": None if visit is None else visit.id,
        "prompt_id": prompt_id,
        "model": "claude-haiku-5-5",
        "input_tokens": 1000,
        "output_tokens": 100,
        "cost_usd": 0.0002,
        "duration_s": 1.25,
        "output": output,
    }
    if visit is not None:
        values["agent_id"] = visit.agent_id
    return LLMCall(**(values | fields))  # pyright: ignore[reportArgumentType]


def _notification(user: User, message: Message, reason: NotificationReason, visit: AgentVisit | None) -> Notification:
    closed = visit is not None
    return Notification(
        user_id=user.id,
        post_id=_POST.id,
        message_id=message.id,
        reason=reason,
        closed_at=_START if closed else None,
        outcome=NotificationOutcome.PASSED if closed else None,
        visit_id=None if visit is None else visit.id,
    )


def _page(
    visits: list[AgentVisit],
    notifications: list[Notification] = [],  # noqa: B006
    calls: list[LLMCall] = [],  # noqa: B006
    searches: list[ToolCall] = [],  # noqa: B006
    research: list[Research] = [],  # noqa: B006
) -> TracePage:
    return trace_page(_TREE, _USERS, visits, notifications, calls, searches, research)


# The summary line of a visit


@pytest.mark.parametrize(
    ("outcome", "fields", "did", "comments", "shown_open"),
    [
        (NotificationOutcome.COMMENTED, {"comment_ids": [_C1.id]}, "commented", [1], False),
        (NotificationOutcome.PASSED, {}, "passed", [], False),
        (NotificationOutcome.PASSED, {"at_research": True}, "passed at research", [], False),
        (NotificationOutcome.LIMIT, {}, "at its limit", [], False),
        (NotificationOutcome.BUDGET, {}, "out of budget", [], False),
        (NotificationOutcome.BUDGET, {"comment_ids": [_C1.id]}, "out of budget after writing", [1], False),
        (NotificationOutcome.FAILED, {}, "failed", [], True),
        (NotificationOutcome.FAILED, {"comment_ids": [_C1.id]}, "failed after writing", [1], True),
        (None, {}, "unfinished (still running, or cut off)", [], True),
    ],
)
def test_a_visit_line_says_what_it_did_as_agents_prints_it(
    outcome: NotificationOutcome | None, fields: dict[str, Any], did: str, comments: list[int], shown_open: bool
) -> None:
    [visit] = _page([_visit(outcome=outcome, reason="Why.", **fields)]).visits

    assert (visit.number, visit.author.name, visit.author.is_agent) == (1, "Developer", True)
    assert (visit.did, visit.comments, visit.reason, visit.open) == (did, comments, "Why.", shown_open)
    assert visit.started_at == _START
    assert visit.duration == (None if outcome is None else "12.3s")


def test_visits_are_numbered_in_the_order_given_and_cost_their_calls() -> None:
    first, second = _visit(), _visit(_SCI)
    calls = [_call(first, "research.v1", None, cost_usd=0.001), _call(first, "decide.v2", None, cost_usd=0.002)]

    page = _page([first, second], calls=calls)

    assert [(v.number, v.author.name, v.cost) for v in page.visits] == [
        (1, "Developer", "$0.003000"),
        (2, "Scientist", "$0.000000"),
    ]


# What a visit handled


def test_each_notification_handled_names_its_place_reason_and_outcome() -> None:
    visit = _visit()
    other = _visit(_SCI)
    notifications = [
        _notification(_DEV, _POST, NotificationReason.NEW_POST, visit),
        _notification(_DEV, _C2, NotificationReason.THREAD, visit),
        _notification(_DEV, _C3, NotificationReason.TOP_LEVEL, visit),
        _notification(_SCI, _C2, NotificationReason.MENTION, other),
    ]
    notifications[1] = notifications[1].model_copy(update={"outcome": NotificationOutcome.LIMIT})

    first, second = _page([visit, other], notifications).visits

    assert [(h.place, h.number, h.reason, h.outcome) for h in first.handled] == [
        ("the post", None, "a new post", "passed"),
        ("#2", 2, "in thread #1", "limit"),
        ("#3", 3, "a new top-level comment", "passed"),
    ]
    assert [(h.place, h.reason) for h in second.handled] == [("#2", "a mention")]


def test_agents_with_pending_notifications_are_waiting_and_a_real_user_is_not() -> None:
    notifications = [
        _notification(_SCI, _POST, NotificationReason.NEW_POST, None),
        _notification(_SCI, _C2, NotificationReason.MENTION, None),
        _notification(_ME, _C1, NotificationReason.TOP_LEVEL, None),
        _notification(_DEV, _POST, NotificationReason.NEW_POST, _visit()),
    ]

    waiting = _page([], notifications).waiting

    assert [(author.name, count) for author, count in waiting] == [("Scientist", 2)]


# Searches and their evidence


def _search(visit: AgentVisit, tool: str, query: str, outcome: ToolCallOutcome, records: int = 0) -> ToolCall:
    error = "rate limit reached" if outcome is ToolCallOutcome.FAILED else None
    return ToolCall(
        visit_id=visit.id,
        agent_id=visit.agent_id,
        post_id=_POST.id,
        tool=tool,
        query=query,
        outcome=outcome,
        evidence=records,
        error=error,
        duration_s=0.84,
    )


def _evidence(title: str) -> Evidence:
    return Evidence(source="pypi", query="q", url=HttpUrl(f"https://example.com/{title}"), title=title, summary="s")


def test_a_search_shows_its_evidence_from_the_saved_research() -> None:
    visit = _visit()
    searches = [
        _search(visit, "pypi", "httpx", ToolCallOutcome.EVIDENCE, 2),
        _search(visit, "pypi", "nope", ToolCallOutcome.NO_INFORMATION),
        _search(visit, "github_search", "httpx", ToolCallOutcome.FAILED),
    ]
    research = Research(
        agent_id=_DEV.id,
        post_id=_POST.id,
        results=[
            ToolResult(tool="pypi", query="httpx", evidence=[_evidence("a"), _evidence("b")]),
            ToolResult(tool="pypi", query="nope"),
            ToolResult(tool="github_search", query="httpx", failed=True),
            ToolResult(tool="github_repo", query=None),
        ],
    )

    [found, nothing, failed] = _page([visit], searches=searches, research=[research]).visits[0].searches

    assert (found.tool, found.query, found.outcome, found.records, found.duration) == (
        "pypi",
        "httpx",
        "evidence",
        2,
        "0.8s",
    )
    assert [(e.title, e.url) for e in found.evidence or []] == [
        ("a", "https://example.com/a"),
        ("b", "https://example.com/b"),
    ]
    assert (nothing.outcome, nothing.evidence) == ("no information", [])
    assert (failed.outcome, failed.error, failed.evidence) == ("failed", "rate limit reached", [])


def test_a_search_whose_research_was_never_saved_has_no_evidence_to_show() -> None:
    visit = _visit(outcome=None)

    [search] = (
        _page([visit], searches=[_search(visit, "pypi", "httpx", ToolCallOutcome.EVIDENCE, 3)]).visits[0].searches
    )

    assert search.evidence is None


def test_the_evidence_is_the_agents_own() -> None:
    visit = _visit(_SCI)
    research = Research(
        agent_id=_DEV.id, post_id=_POST.id, results=[ToolResult(tool="pypi", query="httpx", evidence=[_evidence("a")])]
    )

    [search] = (
        _page([visit], searches=[_search(visit, "pypi", "httpx", ToolCallOutcome.EVIDENCE, 1)], research=[research])
        .visits[0]
        .searches
    )

    assert search.evidence is None


# The answers of the AI calls


def test_a_research_answer_is_shown_as_its_queries_and_reason() -> None:
    visit = _visit()
    plan = ResearchPlan(speak=True, queries={"pypi": ["httpx"], "github_repo": []}, reason="In my area.")

    [call] = _page([visit], calls=[_call(visit, "research.v1", plan.model_dump_json())]).visits[0].calls

    assert call.kind == "research"
    assert call.answer == ResearchAnswer(True, [("pypi", ["httpx"]), ("github_repo", [])], "In my area.")
    assert call.raw == plan.model_dump_json()
    assert (call.prompt_id, call.model, call.input_tokens, call.output_tokens) == (
        "research.v1",
        "claude-haiku-5-5",
        1000,
        100,
    )
    assert (call.cost, call.duration, call.error) == ("$0.000200", "1.2s", None)


@pytest.mark.parametrize(
    ("decision", "places"),
    [
        (
            SpeakDecision(
                reason="New data.", reply_to=[ReplyTo(place=0, point="Mine."), ReplyTo(place=2, point="Yes.")]
            ),
            [("the post", "Mine."), ("#2", "Yes.")],
        ),
        (SpeakDecision(reason="Nothing new.", reply_to=[]), []),
    ],
)
def test_a_decide_answer_is_shown_as_its_reason_and_places(
    decision: SpeakDecision, places: list[tuple[str, str]]
) -> None:
    visit = _visit()

    [call] = _page([visit], calls=[_call(visit, "decide.v2", decision.model_dump_json())]).visits[0].calls

    assert (call.kind, call.answer) == ("decide", DecideAnswer(decision.reason, places))


def test_a_write_answer_links_to_the_comment_it_became() -> None:
    visit = _visit(outcome=NotificationOutcome.COMMENTED, comment_ids=[_C1.id])

    [call] = _page([visit], calls=[_call(visit, "developer.v1", f"  {_C1.text}\n")]).visits[0].calls

    assert (call.kind, call.answer) == ("write", WriteAnswer(1, f"  {_C1.text}\n"))


def test_a_write_answer_that_became_no_comment_is_shown_as_it_is() -> None:
    visit = _visit(outcome=NotificationOutcome.FAILED)

    [call] = _page([visit], calls=[_call(visit, "developer.v1", "   ")]).visits[0].calls

    assert (call.kind, call.answer) == ("write", WriteAnswer(None, "   "))


@pytest.mark.parametrize(
    ("prompt_id", "output", "error"),
    [
        ("decide.v2", '{"reason": "cut', "claude-haiku-5-5 hit max_tokens before finishing the SpeakDecision output"),
        ("developer.v1", "No.", "claude-sonnet-5-5 refused the request (category: cyber)"),
        ("research.v1", None, "Anthropic request timed out"),
    ],
)
def test_a_failed_call_shows_its_error_and_its_answer_only_as_given(
    prompt_id: str, output: str | None, error: str
) -> None:
    visit = _visit(outcome=NotificationOutcome.FAILED)

    [call] = _page([visit], calls=[_call(visit, prompt_id, output, error=error)]).visits[0].calls

    assert (call.kind, call.answer, call.raw, call.error) == ("raw", None, output, error)


# The totals and the earlier calls


def test_the_totals_count_every_call_per_agent_and_for_the_post() -> None:
    dev_visit, sci_visit = _visit(), _visit(_SCI)
    calls = [
        _call(None, "decide.v1", None, cost_usd=0.01),  # stored before visits were
        _call(dev_visit, "research.v1", None, cost_usd=0.02, input_tokens=10, output_tokens=1),
        _call(sci_visit, "research.v1", None, cost_usd=0.04, input_tokens=20, output_tokens=2),
    ]
    searches = [_search(sci_visit, "arxiv", "rust", ToolCallOutcome.EVIDENCE, 3)]

    page = _page([dev_visit, sci_visit], calls=calls, searches=searches)

    assert [
        (t.author and t.author.name, t.visits, t.calls, t.input_tokens, t.output_tokens, t.searches, t.cost)
        for t in page.totals
    ] == [
        ("Developer", 1, 2, 1010, 101, 0, "$0.030000"),
        ("Scientist", 1, 1, 20, 2, 1, "$0.040000"),
        (None, 2, 3, 1030, 103, 1, "$0.070000"),
    ]
    [earlier] = page.earlier
    assert (earlier.author.name, earlier.prompt_id, earlier.cost) == ("Developer", "decide.v1", "$0.010000")


def test_a_post_with_no_trace() -> None:
    page = _page([])

    assert (page.visits, page.earlier, page.waiting) == ([], [], [])
    assert [(t.author, t.calls, t.cost) for t in page.totals] == [(None, 0, "$0.000000")]
    assert (page.post_id, page.start) == (_POST.id, "Is Rust growing?")


# The page


@pytest.fixture
def settings(db_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=str(db_path))  # pyright: ignore[reportCallIssue]


def _get(settings: Settings, path: str) -> str:
    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        response = client.get(path)
    assert response.status_code == 200
    return response.text


class _Input(ToolInput):
    query: str


class _Stub:
    name = "stub"
    description = "Finds things"
    input_model = _Input

    async def run(self, input: _Input) -> list[Evidence]:
        return [
            Evidence(
                source="stub",
                query=input.query,
                url=HttpUrl("https://example.com/repo?a=1&b=2"),
                title="A <b>repo</b>",
                summary="1,234 stars",
            )
        ]


async def _run_developer(settings: Settings, text: str) -> Message:
    """A post, and one visit of a scripted Developer that researches with the stub tool and comments. The other agents
    are notified, but not run, so they are waiting."""
    async with open_session(settings) as session:
        post = await session.forum.create_post(session.me, text)
        role = RoleSpec(
            name="developer",
            display_name="Developer",
            question="Are developers using it?",
            tool_names=("stub",),
            prompt_id="developer.v1",
        )
        llm = FakeLLM(
            rules=[
                (
                    lambda r: r.prompt_id == "research.v1",
                    ResearchPlan(speak=True, queries={"stub": ["rust"]}, reason="My <area>."),
                ),
                (
                    lambda r: r.prompt_id == "decide.v2",
                    SpeakDecision(reason="New data.", reply_to=[ReplyTo(place=0, point="Stars.")]),
                ),
                (lambda r: r.prompt_id == "developer.v1", "It has 1,234 stars."),
            ]
        )
        loop = AgentLoop(
            session.store,
            session.forum,
            session.agents,
            llm=llm,
            roles=RoleRegistry([role]),
            tools=ToolRegistry([_Stub()]),
            settings=settings,
        )
        await loop.run()
        return post


def _text(html: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def test_the_trace_of_a_run_is_shown_and_linked_from_the_post(settings: Settings) -> None:
    post = asyncio.run(_run_developer(settings, "Is Rust growing?"))

    post_html = _get(settings, f"/posts/{post.id}")
    html = _get(settings, f"/posts/{post.id}/trace")

    assert f'<a href="/posts/{post.id}/trace">Trace</a>' in post_html
    assert "<title>Trace · Is Rust growing? · TechDebate</title>" in html
    assert f'Of <a href="/posts/{post.id}">Is Rust growing?</a>' in html
    text = _text(html)
    assert "Waiting: Scientist AI 1" in text  # the Pollster is left out, with no YouTube key
    assert "1 Developer AI commented #1" in text
    assert "the post, a new post: commented" in text
    assert "stub “rust”: evidence, 1 record" in text
    assert "Researched: My &lt;area&gt;." in text
    assert "the post: Stars." in text
    assert f'Wrote <a href="/posts/{post.id}#c1">#1</a>.' in html
    assert '<details class="visit">' in html  # a comment is not open


def test_the_post_total_is_what_the_budget_page_shows(settings: Settings) -> None:
    post = asyncio.run(_run_developer(settings, "Is Rust growing?"))

    async def spent() -> float:
        async with open_store(settings) as store:
            for call in await store.list_llm_calls(post.id):  # priced, as real calls are
                await store.add_llm_call(
                    call.model_copy(update={"id": f"{call.id}x", "cost_usd": 0.0125, "visit_id": None})
                )
            return (await spent_on_post(store, post.id, settings)).spent_usd

    total = asyncio.run(spent())
    html = _get(settings, f"/posts/{post.id}/trace")

    [post_row] = re.findall(r'<tr class="all">(.*?)</tr>', html, re.S)
    assert _text(post_row).endswith(f"${total:.6f}")
    assert total == pytest.approx(0.0375)
    assert "Earlier calls" in html  # the copies have no visit


def test_text_from_models_and_tools_is_escaped_and_evidence_links_are_safe(settings: Settings) -> None:
    post = asyncio.run(_run_developer(settings, "Is <i>Rust</i> growing?"))

    html = _get(settings, f"/posts/{post.id}/trace")

    assert "<b>repo</b>" not in html and "A &lt;b&gt;repo&lt;/b&gt;" in html
    assert "<area>" not in html
    assert "<i>Rust</i>" not in html
    assert '<a href="https://example.com/repo?a=1&amp;b=2" target="_blank" rel="nofollow noopener noreferrer">' in html


def test_a_failed_or_unfinished_visit_is_open(settings: Settings) -> None:
    async def write() -> Message:
        async with open_session(settings) as session:
            post = await session.forum.create_post(session.me, "Is Rust growing?")
            developer = session.agents["developer"]
            await session.store.add_visit(AgentVisit(agent_id=developer.id, post_id=post.id))
            failed = AgentVisit(agent_id=developer.id, post_id=post.id)
            await session.store.add_visit(failed)
            await session.store.finish_visit(
                failed.model_copy(
                    update={"ended_at": datetime.now(UTC), "outcome": NotificationOutcome.FAILED, "reason": "down"}
                )
            )
            return post

    post = asyncio.run(write())
    html = _get(settings, f"/posts/{post.id}/trace")

    assert html.count('<details class="visit attention" open>') == 2
    assert "unfinished (still running, or cut off)" in html
    assert "None closed yet." in html


def test_an_unknown_post_has_no_trace(settings: Settings) -> None:
    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        response = client.get("/posts/nope/trace")

    assert response.status_code == 404
    assert "There is no post here." in response.text
