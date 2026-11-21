"""The Store on SQLite, and opening the database: created on first use and migrated every time it is opened."""

import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import TypeAdapter
from sqlalchemy import Connection, event, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import aliased

from techdebate.config import Settings
from techdebate.errors import StoreError
from techdebate.models import (
    AgentVisit,
    LLMCall,
    Message,
    Notification,
    NotificationOutcome,
    Research,
    ToolCall,
    ToolResult,
    User,
    UserKind,
)
from techdebate.store.base import PostSummary, PostWithComments, Store, StoreOps
from techdebate.store.tables import (
    LLMCallRow,
    MessageRow,
    NotificationRow,
    ResearchRow,
    ToolCallRow,
    UserRow,
    VisitRow,
)

log = logging.getLogger(__name__)

# How long a writer waits for another one to finish. Writes take milliseconds, and AI calls run outside transactions.
_BUSY_TIMEOUT_MS = 5000
# An execution option that makes a transaction start with BEGIN IMMEDIATE, which takes the write lock at once.
_IMMEDIATE = "techdebate_immediate"


def _user(row: UserRow) -> User:
    return User.model_validate(row, from_attributes=True)


def _message(row: MessageRow) -> Message:
    return Message.model_validate(row, from_attributes=True)


def _notification(row: NotificationRow) -> Notification:
    return Notification.model_validate(row, from_attributes=True)


def _llm_call(row: LLMCallRow) -> LLMCall:
    return LLMCall.model_validate(row, from_attributes=True)


def _tool_call(row: ToolCallRow) -> ToolCall:
    return ToolCall.model_validate(row, from_attributes=True)


_COMMENT_IDS = TypeAdapter(list[str])


def _visit(row: VisitRow) -> AgentVisit:
    return AgentVisit(
        id=row.id,
        agent_id=row.agent_id,
        post_id=row.post_id,
        started_at=row.started_at,
        ended_at=row.ended_at,
        outcome=None if row.outcome is None else NotificationOutcome(row.outcome),
        reason=row.reason,
        at_research=row.at_research,
        comment_ids=_COMMENT_IDS.validate_json(row.comment_ids),
    )


_RESULTS = TypeAdapter(list[ToolResult])


def _research(row: ResearchRow) -> Research:
    return Research(
        id=row.id,
        agent_id=row.agent_id,
        post_id=row.post_id,
        results=_RESULTS.validate_json(row.results),
        created_at=row.created_at,
    )


class _SessionOps:
    """The Store's calls, on one session and its transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_user(self, user: User) -> None:
        if await self.find_user(user.username) is not None:
            raise StoreError(f"username {user.username!r} is already taken")
        self._session.add(UserRow(**user.model_dump()))
        await self._session.flush()

    async def get_user(self, user_id: str) -> User | None:
        row = await self._session.get(UserRow, user_id)
        return None if row is None else _user(row)

    async def find_user(self, username: str) -> User | None:
        row = await self._session.scalar(select(UserRow).where(UserRow.username == username))
        return None if row is None else _user(row)

    async def list_users(self, kind: UserKind | None = None) -> list[User]:
        query = select(UserRow).order_by(UserRow.username)
        if kind is not None:
            query = query.where(UserRow.kind == kind)
        return [_user(row) for row in await self._session.scalars(query)]

    async def add_message(self, message: Message) -> None:
        self._session.add(MessageRow(**message.model_dump()))
        await self._session.flush()

    async def get_message(self, message_id: str) -> Message | None:
        row = await self._session.scalar(select(MessageRow).where(MessageRow.id == message_id))
        return None if row is None else _message(row)

    async def get_post(self, post_id: str) -> PostWithComments | None:
        # A comment is saved after its post, so the post comes first.
        query = (
            select(MessageRow)
            .where(or_(MessageRow.id == post_id, MessageRow.post_id == post_id))
            .order_by(MessageRow.seq)
        )
        rows = list(await self._session.scalars(query))
        if not rows or rows[0].id != post_id or rows[0].post_id is not None:
            return None
        return PostWithComments(post=_message(rows[0]), comments=[_message(row) for row in rows[1:]])

    async def list_posts(self) -> list[Message]:
        query = select(MessageRow).where(MessageRow.post_id.is_(None)).order_by(MessageRow.seq.desc())
        return [_message(row) for row in await self._session.scalars(query)]

    async def list_post_summaries(self) -> list[PostSummary]:
        comment = aliased(MessageRow)
        counts = (
            select(comment.post_id, func.count().label("comments"))
            .where(comment.post_id.is_not(None))
            .group_by(comment.post_id)
            .subquery()
        )
        query = (
            select(MessageRow, func.coalesce(counts.c.comments, 0))
            .outerjoin(counts, counts.c.post_id == MessageRow.id)
            .where(MessageRow.post_id.is_(None))
            .order_by(MessageRow.seq.desc())
        )
        return [PostSummary(post=_message(row), comments=count) for row, count in await self._session.execute(query)]

    async def add_notifications(self, notifications: Sequence[Notification]) -> None:
        self._session.add_all([NotificationRow(**notification.model_dump()) for notification in notifications])
        await self._session.flush()

    async def list_notifications(self, user_id: str, *, include_closed: bool = False) -> list[Notification]:
        query = select(NotificationRow).where(NotificationRow.user_id == user_id).order_by(NotificationRow.seq)
        if not include_closed:
            query = query.where(NotificationRow.closed_at.is_(None))
        return [_notification(row) for row in await self._session.scalars(query)]

    async def close_notifications(
        self, notification_ids: Sequence[str], outcome: NotificationOutcome, *, visit_id: str | None = None
    ) -> None:
        if not notification_ids:
            return
        await self._session.execute(
            update(NotificationRow)
            .where(NotificationRow.id.in_(notification_ids), NotificationRow.closed_at.is_(None))
            .values(closed_at=datetime.now(UTC), outcome=outcome, visit_id=visit_id)
        )

    async def add_visit(self, visit: AgentVisit) -> None:
        comment_ids = _COMMENT_IDS.dump_json(visit.comment_ids).decode()
        self._session.add(VisitRow(**visit.model_dump(exclude={"comment_ids"}), comment_ids=comment_ids))
        await self._session.flush()

    async def finish_visit(self, visit: AgentVisit) -> None:
        await self._session.execute(
            update(VisitRow)
            .where(VisitRow.id == visit.id)
            .values(
                ended_at=visit.ended_at,
                outcome=visit.outcome,
                reason=visit.reason,
                at_research=visit.at_research,
                comment_ids=_COMMENT_IDS.dump_json(visit.comment_ids).decode(),
            )
        )

    async def list_visits(self, post_id: str) -> list[AgentVisit]:
        query = select(VisitRow).where(VisitRow.post_id == post_id).order_by(VisitRow.seq)
        return [_visit(row) for row in await self._session.scalars(query)]

    async def add_llm_call(self, call: LLMCall) -> None:
        self._session.add(LLMCallRow(**call.model_dump()))
        await self._session.flush()

    async def list_llm_calls(self, post_id: str) -> list[LLMCall]:
        query = select(LLMCallRow).where(LLMCallRow.post_id == post_id).order_by(LLMCallRow.seq)
        return [_llm_call(row) for row in await self._session.scalars(query)]

    async def add_tool_call(self, call: ToolCall) -> None:
        self._session.add(ToolCallRow(**call.model_dump()))
        await self._session.flush()

    async def list_tool_calls(self, post_id: str) -> list[ToolCall]:
        query = select(ToolCallRow).where(ToolCallRow.post_id == post_id).order_by(ToolCallRow.seq)
        return [_tool_call(row) for row in await self._session.scalars(query)]

    async def spent_usd(
        self, *, post_id: str | None = None, agent_id: str | None = None, since: datetime | None = None
    ) -> float:
        query = select(func.coalesce(func.sum(LLMCallRow.cost_usd), 0.0))
        if post_id is not None:
            query = query.where(LLMCallRow.post_id == post_id)
        if agent_id is not None:
            query = query.where(LLMCallRow.agent_id == agent_id)
        if since is not None:
            query = query.where(LLMCallRow.created_at >= since)
        return float(await self._session.scalar(query) or 0.0)

    async def add_research(self, research: Research) -> None:
        results = _RESULTS.dump_json(research.results).decode()
        self._session.add(ResearchRow(**research.model_dump(exclude={"results"}), results=results))
        await self._session.flush()

    async def get_research(self, agent_id: str, post_id: str) -> Research | None:
        query = select(ResearchRow).where(ResearchRow.agent_id == agent_id, ResearchRow.post_id == post_id)
        row = await self._session.scalar(query)
        return None if row is None else _research(row)


class SqliteStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._reads = async_sessionmaker(engine, expire_on_commit=False)
        self._writes = async_sessionmaker(engine, expire_on_commit=False, execution_options={_IMMEDIATE: True})

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[StoreOps]:
        async with self._writes() as session, session.begin():
            yield _SessionOps(session)

    @asynccontextmanager
    async def _read(self) -> AsyncGenerator[StoreOps]:
        async with self._reads() as session, session.begin():
            yield _SessionOps(session)

    async def add_user(self, user: User) -> None:
        async with self.transaction() as tx:
            await tx.add_user(user)

    async def get_user(self, user_id: str) -> User | None:
        async with self._read() as ops:
            return await ops.get_user(user_id)

    async def find_user(self, username: str) -> User | None:
        async with self._read() as ops:
            return await ops.find_user(username)

    async def list_users(self, kind: UserKind | None = None) -> list[User]:
        async with self._read() as ops:
            return await ops.list_users(kind)

    async def add_message(self, message: Message) -> None:
        async with self.transaction() as tx:
            await tx.add_message(message)

    async def get_message(self, message_id: str) -> Message | None:
        async with self._read() as ops:
            return await ops.get_message(message_id)

    async def get_post(self, post_id: str) -> PostWithComments | None:
        async with self._read() as ops:
            return await ops.get_post(post_id)

    async def list_posts(self) -> list[Message]:
        async with self._read() as ops:
            return await ops.list_posts()

    async def list_post_summaries(self) -> list[PostSummary]:
        async with self._read() as ops:
            return await ops.list_post_summaries()

    async def add_notifications(self, notifications: Sequence[Notification]) -> None:
        async with self.transaction() as tx:
            await tx.add_notifications(notifications)

    async def list_notifications(self, user_id: str, *, include_closed: bool = False) -> list[Notification]:
        async with self._read() as ops:
            return await ops.list_notifications(user_id, include_closed=include_closed)

    async def close_notifications(
        self, notification_ids: Sequence[str], outcome: NotificationOutcome, *, visit_id: str | None = None
    ) -> None:
        async with self.transaction() as tx:
            await tx.close_notifications(notification_ids, outcome, visit_id=visit_id)

    async def add_visit(self, visit: AgentVisit) -> None:
        async with self.transaction() as tx:
            await tx.add_visit(visit)

    async def finish_visit(self, visit: AgentVisit) -> None:
        async with self.transaction() as tx:
            await tx.finish_visit(visit)

    async def list_visits(self, post_id: str) -> list[AgentVisit]:
        async with self._read() as ops:
            return await ops.list_visits(post_id)

    async def add_llm_call(self, call: LLMCall) -> None:
        async with self.transaction() as tx:
            await tx.add_llm_call(call)

    async def list_llm_calls(self, post_id: str) -> list[LLMCall]:
        async with self._read() as ops:
            return await ops.list_llm_calls(post_id)

    async def add_tool_call(self, call: ToolCall) -> None:
        async with self.transaction() as tx:
            await tx.add_tool_call(call)

    async def list_tool_calls(self, post_id: str) -> list[ToolCall]:
        async with self._read() as ops:
            return await ops.list_tool_calls(post_id)

    async def spent_usd(
        self, *, post_id: str | None = None, agent_id: str | None = None, since: datetime | None = None
    ) -> float:
        async with self._read() as ops:
            return await ops.spent_usd(post_id=post_id, agent_id=agent_id, since=since)

    async def add_research(self, research: Research) -> None:
        async with self.transaction() as tx:
            await tx.add_research(research)

    async def get_research(self, agent_id: str, post_id: str) -> Research | None:
        async with self._read() as ops:
            return await ops.get_research(agent_id, post_id)


def _on_connect(dbapi_connection: Any, _record: Any) -> None:
    # The driver would start transactions on its own, and skip them before reads; _on_begin starts them instead.
    dbapi_connection.isolation_level = None
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")  # SQLite ignores foreign keys unless asked, on every connection
    cursor.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    cursor.close()


def _on_begin(connection: Connection) -> None:
    connection.exec_driver_sql("BEGIN IMMEDIATE" if connection.get_execution_options().get(_IMMEDIATE) else "BEGIN")


def _create_engine(path: str) -> AsyncEngine:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    event.listen(engine.sync_engine, "connect", _on_connect)
    event.listen(engine.sync_engine, "begin", _on_begin)
    return engine


def _alembic_config(connection: Connection | None = None) -> Config:
    """The migrations ship inside the package. With a connection, they run on it, in its transaction."""
    config = Config()
    config.set_main_option("script_location", str(files("techdebate.store").joinpath("migrations")))
    config.attributes["connection"] = connection
    return config


def _upgrade(connection: Connection) -> str | None:
    """The database file's full path when the database was new (it had no tables at all), else None."""
    config = _alembic_config(connection)
    known = {script.revision for script in ScriptDirectory.from_config(config).walk_revisions()}
    current = MigrationContext.configure(connection).get_current_revision()
    if current is not None and current not in known:
        raise StoreError(f"the database is at revision {current}, which this version of techdebate does not know")
    command.upgrade(config, "head")
    if current is not None:
        return None
    # SQLite reports the full path of each file it opened, as rows of (seq, name, file); ours is "main".
    return next(row[2] for row in connection.exec_driver_sql("PRAGMA database_list") if row[1] == "main")


async def _migrate(engine: AsyncEngine) -> str | None:
    # One write transaction: two processes opening the database at the same time migrate it once.
    async with engine.connect() as connection:
        await connection.execution_options(**{_IMMEDIATE: True})
        async with connection.begin():
            return await connection.run_sync(_upgrade)


@asynccontextmanager
async def open_store(settings: Settings) -> AsyncGenerator[Store]:
    """The database at db_path, created if it does not exist and migrated to the newest version."""
    engine = _create_engine(settings.db_path)
    try:
        if created := await _migrate(engine):
            log.info("created database path=%s", created)
        yield SqliteStore(engine)
    finally:
        await engine.dispose()
