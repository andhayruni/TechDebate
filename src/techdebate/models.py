"""Core records shared by agents, the orchestrator, storage and the web page."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


def _new_id() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Evidence(BaseModel):
    """A fact fetched from a public source by one of the data tools."""

    id: str = Field(default_factory=_new_id)
    source: str  # the tool name, e.g. "github_search", "github_repo", "pypi", "arxiv", "youtube"
    query: str
    url: HttpUrl
    title: str
    summary: str
    value: float | None = None
    fetched_at: datetime = Field(default_factory=_now)


class ToolResult(BaseModel):
    """What one query found. With no evidence, the query found no information, or the tool failed."""

    model_config = ConfigDict(frozen=True)

    tool: str
    query: str | None  # None: the tool was not used
    evidence: list[Evidence] = Field(default_factory=list[Evidence])
    failed: bool = False


class ResearchPlan(BaseModel):
    """An agent's first look at a post: whether it is in the agent's area, and what to search for. The model fills this
    in, so every field is part of the schema it sees."""

    speak: bool
    queries: dict[str, list[str]] = Field(default_factory=dict[str, list[str]])  # per tool; empty: the tool doesn't fit
    reason: str


class ReplyTo(BaseModel):
    """One place an agent comments on, with the point its comment there makes."""

    place: int  # 0 is the post itself, and n is comment #n
    point: str


class SpeakDecision(BaseModel):
    """Where an agent comments, after its research. The model fills this in from choices built for each call."""

    # First, so the model gives its reason before it picks the places.
    reason: str
    reply_to: list[ReplyTo]  # empty: a pass


class UserKind(StrEnum):
    AGENT = "agent"
    REAL = "real"


# Every username, of either kind: 1 to 30 lowercase letters, digits and underscores.
USERNAME_PATTERN = r"^[a-z0-9_]{1,30}$"


class User(BaseModel):
    """Anyone who writes: an AI agent or a real user. Both kinds share one set of usernames."""

    id: str = Field(default_factory=_new_id)
    username: str = Field(pattern=USERNAME_PATTERN)  # an agent's username is its role's name
    kind: UserKind
    created_at: datetime = Field(default_factory=_now)


class Message(BaseModel):
    """A post or a comment. A post has no parent and no post. A comment answers its parent (the post or another
    comment), and post_id is the post at the root of its tree, so all of a post's comments are found in one lookup."""

    id: str = Field(default_factory=_new_id)
    post_id: str | None = None
    parent_id: str | None = None
    author_id: str
    text: str
    created_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def _post_or_comment(self) -> Self:
        if (self.post_id is None) != (self.parent_id is None):
            raise ValueError("a post has neither post_id nor parent_id, and a comment has both")
        return self


class NotificationReason(StrEnum):
    NEW_POST = "new_post"  # a new post, to every agent
    THREAD = "thread"  # a new comment, to everyone who has commented in its thread
    TOP_LEVEL = "top_level"  # a new top-level comment, to the post's author and the agents that commented on the post
    MENTION = "mention"  # an @username written by a real user


class NotificationOutcome(StrEnum):
    """How a notification was closed: listed by a real user, or what an agent's visit did with it."""

    SEEN = "seen"  # a real user listed it
    COMMENTED = "commented"  # the agent wrote at least one comment
    PASSED = "passed"  # the agent looked, and had nothing new to add
    LIMIT = "limit"  # its thread is full for the agent, or no place on the post is open to it
    FAILED = "failed"  # an AI call failed during the visit
    BUDGET = "budget"  # a budget that covers the agent's calls on the post is used up


class Notification(BaseModel):
    """Tells a user about a message. It is pending until it is closed, and closing sets closed_at and outcome
    together."""

    id: str = Field(default_factory=_new_id)
    user_id: str  # who is notified
    post_id: str
    message_id: str  # the message that caused it
    reason: NotificationReason
    created_at: datetime = Field(default_factory=_now)
    closed_at: datetime | None = None
    outcome: NotificationOutcome | None = None
    visit_id: str | None = None  # the agent's visit that closed it; None while pending, and for a real user's

    @property
    def pending(self) -> bool:
        return self.closed_at is None

    @model_validator(mode="after")
    def _closed_with_outcome(self) -> Self:
        if (self.closed_at is None) != (self.outcome is None):
            raise ValueError("a notification has both closed_at and outcome, or neither")
        return self


class AgentVisit(BaseModel):
    """One visit of an agent to a post, stored from its start. A visit that was cut off has no end and no outcome."""

    id: str = Field(default_factory=_new_id)
    agent_id: str
    post_id: str
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    outcome: NotificationOutcome | None = None  # what the visit did with the notifications it acted on
    reason: str = ""  # the reason of the call that ended it, what was full, or the error
    at_research: bool = False  # a pass at the research call, not at the decide call
    comment_ids: list[str] = Field(default_factory=list[str])  # the comments it wrote, in the order it wrote them

    @property
    def finished(self) -> bool:
        return self.ended_at is not None


class LLMCall(BaseModel):
    """One AI call an agent made on a post: as it returned, or as it failed."""

    id: str = Field(default_factory=_new_id)
    agent_id: str
    post_id: str
    visit_id: str | None = None  # None only for the calls stored before visits were
    prompt_id: str  # which call it was, e.g. "research.v1"
    model: str  # the model that answered, or the one requested when none did
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_s: float  # the whole call, retries included
    output: str | None = None  # the text of the answer: JSON for a call with a schema. None: no answer
    error: str | None = None  # why the call failed. None: it returned
    created_at: datetime = Field(default_factory=_now)  # when the call returned or failed


class ToolCallOutcome(StrEnum):
    EVIDENCE = "evidence"  # the tool returned evidence
    NO_INFORMATION = "no_information"  # it returned nothing, or found nothing under that name
    FAILED = "failed"  # it raised


class ToolCall(BaseModel):
    """One search an agent made in a visit. Its evidence is in the agent's research on the post."""

    id: str = Field(default_factory=_new_id)
    visit_id: str
    agent_id: str
    post_id: str
    tool: str
    query: str
    outcome: ToolCallOutcome
    evidence: int  # how many records came back
    error: str | None = None  # the message of what it raised
    duration_s: float
    created_at: datetime = Field(default_factory=_now)  # when the search ended


class Research(BaseModel):
    """What an agent found when it researched a post: one per agent and post, so it never searches a post twice."""

    id: str = Field(default_factory=_new_id)
    agent_id: str
    post_id: str
    results: list[ToolResult]  # in the order gather returns them, including the tools not used
    created_at: datetime = Field(default_factory=_now)
