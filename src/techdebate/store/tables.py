"""The database tables. They stay inside the store package: the Store takes and returns the records in models.py.

`messages` and `notifications` are numbered in the order they are saved (`seq`), so "oldest first" is the same on every
read, even for rows saved in the same moment. The uuid `id` stays the ID everything else uses.
"""

from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import DateTime, ForeignKey, Index, MetaData, Text, UniqueConstraint
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator, TypeEngine

# Named constraints, so that later migrations can change them (SQLite rebuilds the table to do it).
_NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class UTCDateTime(TypeDecorator[datetime]):
    """A datetime stored as UTC without a zone, since SQLite keeps none, and read back as UTC. A datetime without a
    zone is refused rather than guessed."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"datetime without a time zone: {value.isoformat()}")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=_NAMING_CONVENTION)
    type_annotation_map: ClassVar[dict[Any, TypeEngine[Any]]] = {datetime: UTCDateTime()}


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)  # one set of names for agents and real users
    kind: Mapped[str]
    created_at: Mapped[datetime]


class MessageRow(Base):
    __tablename__ = "messages"

    seq: Mapped[int] = mapped_column(primary_key=True)
    id: Mapped[str] = mapped_column(unique=True)
    post_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id"), index=True)  # all of a post's comments
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id"))
    author_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]


class NotificationRow(Base):
    __tablename__ = "notifications"
    __table_args__ = (Index(None, "user_id", "closed_at"),)  # a user's pending notifications

    seq: Mapped[int] = mapped_column(primary_key=True)
    id: Mapped[str] = mapped_column(unique=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    post_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    reason: Mapped[str]
    created_at: Mapped[datetime]
    closed_at: Mapped[datetime | None]
    outcome: Mapped[str | None]


class LLMCallRow(Base):
    __tablename__ = "llm_calls"

    seq: Mapped[int] = mapped_column(primary_key=True)
    id: Mapped[str] = mapped_column(unique=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    post_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), index=True)  # a post's calls
    prompt_id: Mapped[str]
    model: Mapped[str]
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    cost_usd: Mapped[float]
    duration_s: Mapped[float]
    created_at: Mapped[datetime]


class ResearchRow(Base):
    """One per agent and post. The results are JSON text: they are only ever read back whole."""

    __tablename__ = "research"
    __table_args__ = (UniqueConstraint("agent_id", "post_id"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    post_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    results: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]
