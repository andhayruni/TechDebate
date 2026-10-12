"""Core records shared by agents, the orchestrator, storage and the web page."""

from datetime import UTC, date, datetime
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
    source: str  # e.g. "github", "pypi", "arxiv", "hackernews"
    query: str
    url: HttpUrl
    title: str
    summary: str
    value: float | None = None
    fetched_at: datetime = Field(default_factory=_now)


class Claim(BaseModel):
    """One checkable statement inside a turn. Every claim should cite a source."""

    statement: str
    value: float | None = None
    unit: str | None = None
    source_url: HttpUrl | None = None
    source_date: date | None = None

    @property
    def is_cited(self) -> bool:
        return self.source_url is not None


class TurnKind(StrEnum):
    OPENING = "opening"
    REBUTTAL = "rebuttal"


class Turn(BaseModel):
    id: str = Field(default_factory=_new_id)
    analyst: str
    round: int
    kind: TurnKind
    text: str
    claims: list[Claim] = Field(default_factory=list[Claim])
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)


class RelevanceDecision(BaseModel):
    analyst: str
    join: bool
    reason: str


class DebateStatus(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


class Debate(BaseModel):
    id: str = Field(default_factory=_new_id)
    topic: str
    status: DebateStatus = DebateStatus.RUNNING
    decisions: list[RelevanceDecision] = Field(default_factory=list[RelevanceDecision])
    evidence: list[Evidence] = Field(default_factory=list[Evidence])
    turns: list[Turn] = Field(default_factory=list[Turn])
    total_cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)

    @property
    def participants(self) -> list[str]:
        return [d.analyst for d in self.decisions if d.join]
