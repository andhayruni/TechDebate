"""Core records shared by agents, the orchestrator, storage and the web page."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl


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


class Turn(BaseModel):
    id: str = Field(default_factory=_new_id)
    analyst: str
    text: str
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)


class DebateStatus(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


class Debate(BaseModel):
    id: str = Field(default_factory=_new_id)
    topic: str
    status: DebateStatus = DebateStatus.RUNNING
    turns: list[Turn] = Field(default_factory=list[Turn])
    total_cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)
