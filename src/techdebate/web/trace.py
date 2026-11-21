"""What the trace page shows: a post's visits in the order they started, each with the notifications it handled, its
searches with what they found, and its AI calls with their answers, then the totals. It is built here from the stored
records (8.1), and the template only lays it out.

Everything shown from a model or a tool is plain text: the template escapes it, and none of it is rendered as
markdown."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import ValidationError

from techdebate.models import (
    AgentVisit,
    Evidence,
    LLMCall,
    Message,
    Notification,
    NotificationOutcome,
    NotificationReason,
    Research,
    ResearchPlan,
    SpeakDecision,
    ToolCall,
    ToolCallOutcome,
    User,
    UserKind,
)
from techdebate.places import POST
from techdebate.tree import Tree
from techdebate.web.pages import TITLE_CHARS, Author, author_of, start_of

# What each notification's reason means, in the words of the page.
_REASONS = {
    NotificationReason.NEW_POST: "a new post",
    NotificationReason.THREAD: "in thread #{thread}",
    NotificationReason.TOP_LEVEL: "a new top-level comment",
    NotificationReason.MENTION: "a mention",
}

_SEARCH_OUTCOMES = {
    ToolCallOutcome.EVIDENCE: "evidence",
    ToolCallOutcome.NO_INFORMATION: "no information",
    ToolCallOutcome.FAILED: "failed",
}

# What a visit did, as `agents` prints it (5.6). Comments it wrote follow.
_DID = {
    NotificationOutcome.COMMENTED: "commented",
    NotificationOutcome.PASSED: "passed",
    NotificationOutcome.LIMIT: "at its limit",
    NotificationOutcome.FAILED: "failed",
    NotificationOutcome.BUDGET: "out of budget",
}


@dataclass(frozen=True)
class Handled:
    """A notification a visit closed: the message it was about, why the agent was told, and what came of it."""

    place: str  # "the post" or "#4"
    number: int | None  # the comment's number, to link to it; None for the post
    reason: str
    outcome: str


@dataclass(frozen=True)
class EvidenceView:
    title: str
    summary: str
    url: str


@dataclass(frozen=True)
class SearchView:
    tool: str
    query: str
    outcome: str
    records: int
    duration: str
    error: str | None
    evidence: list[EvidenceView] | None  # from the saved research; None when it was never saved


@dataclass(frozen=True)
class ResearchAnswer:
    researched: bool
    queries: list[tuple[str, list[str]]]  # per tool, in the role's order; empty: not used
    reason: str


@dataclass(frozen=True)
class DecideAnswer:
    reason: str
    places: list[tuple[str, str]]  # the place ("the post" or "#4") and its point; empty: a pass


@dataclass(frozen=True)
class WriteAnswer:
    comment: int | None  # the comment it became, when it was saved
    text: str  # the answer, shown when it did not become a comment


@dataclass(frozen=True)
class CallView:
    author: Author
    prompt_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: str
    duration: str
    error: str | None
    kind: Literal["research", "decide", "write", "raw"]  # how `answer` reads; "raw": only as given
    answer: ResearchAnswer | DecideAnswer | WriteAnswer | None
    raw: str | None  # the answer as the model gave it; None: no answer


@dataclass(frozen=True)
class VisitView:
    number: int
    author: Author
    started_at: datetime
    duration: str | None  # None: unfinished
    did: str  # "commented", "passed at research", "unfinished", ...
    comments: list[int]  # the comments it wrote, by number
    reason: str
    cost: str
    open: bool  # failed or unfinished: shown open
    handled: list[Handled]
    searches: list[SearchView]
    calls: list[CallView]


@dataclass(frozen=True)
class Total:
    author: Author | None  # None: the whole post
    visits: int
    calls: int
    input_tokens: int
    output_tokens: int
    searches: int
    cost: str


@dataclass(frozen=True)
class TracePage:
    post_id: str
    start: str  # the start of the post, as plain text
    waiting: list[tuple[Author, int]]  # agents with notifications still pending on the post, oldest first
    totals: list[Total]  # each agent that did anything, then the post
    visits: list[VisitView]
    earlier: list[CallView]  # AI calls stored before visits were


def trace_page(
    tree: Tree,
    users: Mapping[str, User],
    visits: Sequence[AgentVisit],
    notifications: Sequence[Notification],
    calls: Sequence[LLMCall],
    searches: Sequence[ToolCall],
    research: Iterable[Research],
) -> TracePage:
    """The trace of a post. `users` maps each user's ID to its user, `visits` are in the order they started, and
    `research` is what the agents saved on the post."""
    saved = {(r.agent_id, result.tool, result.query): result.evidence for r in research for result in r.results}
    view_visits = [
        _visit_view(
            n,
            visit,
            tree,
            users,
            [x for x in notifications if x.visit_id == visit.id],
            [c for c in calls if c.visit_id == visit.id],
            [s for s in searches if s.visit_id == visit.id],
            saved,
        )
        for n, visit in enumerate(visits, start=1)
    ]
    pending = Counter(n.user_id for n in notifications if n.pending and users[n.user_id].kind is UserKind.AGENT)
    return TracePage(
        post_id=tree.post.id,
        start=start_of(tree.post.text, TITLE_CHARS),
        waiting=[(author_of(users[agent_id]), count) for agent_id, count in pending.items()],
        totals=_totals(users, visits, calls, searches),
        visits=view_visits,
        earlier=[_call_view(call, tree, users, ()) for call in calls if call.visit_id is None],
    )


def _visit_view(
    number: int,
    visit: AgentVisit,
    tree: Tree,
    users: Mapping[str, User],
    handled: Sequence[Notification],
    calls: Sequence[LLMCall],
    searches: Sequence[ToolCall],
    saved: Mapping[tuple[str, str, str | None], list[Evidence]],
) -> VisitView:
    written = [c for c in tree.comments if c.id in set(visit.comment_ids)]
    if visit.outcome is None:
        did = "unfinished (still running, or cut off)"
    else:
        did = _DID[visit.outcome] + (" at research" if visit.at_research else "")
        if written and visit.outcome is not NotificationOutcome.COMMENTED:
            did += " after writing"
    duration = None if visit.ended_at is None else _seconds((visit.ended_at - visit.started_at).total_seconds())
    return VisitView(
        number=number,
        author=author_of(users[visit.agent_id]),
        started_at=visit.started_at,
        duration=duration,
        did=did,
        comments=[tree.number(c) for c in written],
        reason=visit.reason,
        cost=_usd(sum(c.cost_usd for c in calls)),
        open=visit.outcome in (None, NotificationOutcome.FAILED),
        handled=[_handled(n, tree) for n in handled],
        searches=[_search_view(s, saved) for s in searches],
        calls=[_call_view(c, tree, users, written) for c in calls],
    )


def _handled(notification: Notification, tree: Tree) -> Handled:
    if notification.message_id == tree.post.id:
        place, number, thread = "the post", None, None
    else:
        comment = next(c for c in tree.comments if c.id == notification.message_id)
        number = tree.number(comment)
        place, thread = f"#{number}", tree.number(tree.top_level(comment))
    return Handled(
        place=place,
        number=number,
        reason=_REASONS[notification.reason].format(thread=thread),
        outcome=str(notification.outcome or "pending"),
    )


def _search_view(search: ToolCall, saved: Mapping[tuple[str, str, str | None], list[Evidence]]) -> SearchView:
    key = (search.agent_id, search.tool, search.query)
    found = saved.get(key)
    return SearchView(
        tool=search.tool,
        query=search.query,
        outcome=_SEARCH_OUTCOMES[search.outcome],
        records=search.evidence,
        duration=_seconds(search.duration_s),
        error=search.error,
        evidence=None if found is None else [EvidenceView(e.title, e.summary, str(e.url)) for e in found],
    )


def _call_view(call: LLMCall, tree: Tree, users: Mapping[str, User], written: Sequence[Message]) -> CallView:
    kind, answer = _answer(call, tree, written)
    return CallView(
        author=author_of(users[call.agent_id]),
        prompt_id=call.prompt_id,
        model=call.model,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        cost=_usd(call.cost_usd),
        duration=_seconds(call.duration_s),
        error=call.error,
        kind=kind,
        answer=answer,
        raw=call.output,
    )


def _answer(
    call: LLMCall, tree: Tree, written: Sequence[Message]
) -> tuple[Literal["research", "decide", "write", "raw"], ResearchAnswer | DecideAnswer | WriteAnswer | None]:
    """The answer in a readable form, by which call it was: the research and decide calls are shared by every agent,
    and any other prompt is an agent's write prompt. An answer that does not parse is shown only as given."""
    if call.output is None:
        return "raw", None
    name = call.prompt_id.split(".")[0]
    try:
        if name == "research":
            plan = ResearchPlan.model_validate_json(call.output)
            return "research", ResearchAnswer(plan.speak, list(plan.queries.items()), plan.reason)
        if name == "decide":
            decision = SpeakDecision.model_validate_json(call.output)
            places = [(_place(item.place), item.point) for item in decision.reply_to]
            return "decide", DecideAnswer(decision.reason, places)
    except ValidationError:
        return "raw", None
    if call.error is not None:  # a write call that failed: its answer is not a comment
        return "raw", None
    # The forum saves a comment without its surrounding whitespace (4.3).
    comment = next((c for c in written if c.text == call.output.strip()), None)
    return "write", WriteAnswer(None if comment is None else tree.number(comment), call.output)


def _totals(
    users: Mapping[str, User], visits: Sequence[AgentVisit], calls: Sequence[LLMCall], searches: Sequence[ToolCall]
) -> list[Total]:
    """One row per agent that made a visit or a call, in the order they first did, then the post. Every call counts,
    those from before visits were stored too, so the post's cost is what the budget page shows for it."""
    agents = list(dict.fromkeys([v.agent_id for v in visits] + [c.agent_id for c in calls]))

    def total(author: Author | None, agent_id: str | None) -> Total:
        def mine(row_agent: str) -> bool:
            return agent_id is None or row_agent == agent_id

        its_calls = [c for c in calls if mine(c.agent_id)]
        return Total(
            author=author,
            visits=sum(mine(v.agent_id) for v in visits),
            calls=len(its_calls),
            input_tokens=sum(c.input_tokens for c in its_calls),
            output_tokens=sum(c.output_tokens for c in its_calls),
            searches=sum(mine(s.agent_id) for s in searches),
            cost=_usd(sum(c.cost_usd for c in its_calls)),
        )

    return [total(author_of(users[a]), a) for a in agents] + [total(None, None)]


def _place(place: int) -> str:
    return "the post" if place == POST else f"#{place}"


def _usd(amount: float) -> str:
    return f"${amount:.6f}"


def _seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"
