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
    """How a notification was closed. The agents' outcomes are added with the agent loop."""

    SEEN = "seen"  # a real user listed it


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

    @property
    def pending(self) -> bool:
        return self.closed_at is None

    @model_validator(mode="after")
    def _closed_with_outcome(self) -> Self:
        if (self.closed_at is None) != (self.outcome is None):
            raise ValueError("a notification has both closed_at and outcome, or neither")
        return self


class LLMCall(BaseModel):
    """One AI call an agent made on a post, as it returned."""

    id: str = Field(default_factory=_new_id)
    agent_id: str
    post_id: str
    prompt_id: str  # which call it was, e.g. "research.v1"
    model: str  # the model that answered
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_s: float  # the whole call, retries included
    created_at: datetime = Field(default_factory=_now)  # when the call returned


class Research(BaseModel):
    """What an agent found when it researched a post: one per agent and post, so it never searches a post twice."""

    id: str = Field(default_factory=_new_id)
    agent_id: str
    post_id: str
    results: list[ToolResult]  # in the order gather returns them, including the tools not used
    created_at: datetime = Field(default_factory=_now)
