import asyncio
import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import IntegrityError, StatementError

from techdebate.config import Settings
from techdebate.errors import StoreError
from techdebate.models import Message, Notification, NotificationOutcome, NotificationReason, User, UserKind
from techdebate.store import Store, open_store
from techdebate.store.tables import Base


def _settings(path: Path) -> Settings:
    return Settings(_env_file=None, db_path=str(path))  # pyright: ignore[reportCallIssue]


@pytest.fixture
def selects() -> Iterator[list[str]]:
    """The SELECT statements sent to any database while the test runs."""
    seen: list[str] = []

    def record(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            seen.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    yield seen
    event.remove(Engine, "before_cursor_execute", record)


async def _users(store: Store) -> tuple[User, User]:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    await store.add_user(me)
    await store.add_user(developer)
    return me, developer


def _comment(parent: Message, author: User, text: str) -> Message:
    return Message(post_id=parent.post_id or parent.id, parent_id=parent.id, author_id=author.id, text=text)


async def test_users_round_trip(store: Store) -> None:
    me, developer = await _users(store)

    assert await store.get_user(me.id) == me
    assert await store.find_user("developer") == developer
    assert await store.find_user("nobody") is None
    assert await store.list_users() == [developer, me]
    assert await store.list_users(UserKind.AGENT) == [developer]


@pytest.mark.parametrize("kind", list(UserKind))
async def test_a_username_is_taken_across_both_kinds(store: Store, kind: UserKind) -> None:
    await _users(store)

    with pytest.raises(StoreError, match="username 'developer' is already taken"):
        await store.add_user(User(username="developer", kind=kind))


async def test_messages_round_trip(store: Store) -> None:
    me, developer = await _users(store)
    post = Message(author_id=me.id, text="Is Rust growing?")
    comment = _comment(post, developer, "Yes.")
    await store.add_message(post)
    await store.add_message(comment)

    assert await store.get_message(post.id) == post
    assert await store.get_message(comment.id) == comment
    assert await store.get_message("unknown") is None


async def test_a_post_comes_with_all_its_comments_in_one_lookup_and_in_insertion_order(
    store: Store, selects: list[str]
) -> None:
    me, developer = await _users(store)
    post = Message(author_id=me.id, text="Is Rust growing?")
    other = Message(author_id=me.id, text="Is Go growing?")
    same_moment = datetime(2026, 11, 11, tzinfo=UTC)
    # Saved in this order, all at the same moment, and replies three levels deep.
    first = _comment(post, developer, "first").model_copy(update={"created_at": same_moment})
    second = _comment(first, me, "second").model_copy(update={"created_at": same_moment})
    third = _comment(second, developer, "third").model_copy(update={"created_at": same_moment})
    fourth = _comment(post, me, "fourth").model_copy(update={"created_at": same_moment})
    for message in (post, other, first, _comment(other, me, "elsewhere"), second, third, fourth):
        await store.add_message(message)

    selects.clear()
    found = await store.get_post(post.id)

    assert found is not None
    assert found.post == post
    assert [c.text for c in found.comments] == ["first", "second", "third", "fourth"]
    assert len(selects) == 1


async def test_get_post_is_none_for_a_comment_or_an_unknown_id(store: Store) -> None:
    me, developer = await _users(store)
    post = Message(author_id=me.id, text="Is Rust growing?")
    comment = _comment(post, developer, "Yes.")
    await store.add_message(post)
    await store.add_message(comment)

    assert await store.get_post(comment.id) is None
    assert await store.get_post("unknown") is None


async def test_list_posts_newest_first_without_comments(store: Store) -> None:
    me, developer = await _users(store)
    older = Message(author_id=me.id, text="older")
    newer = Message(author_id=me.id, text="newer").model_copy(update={"created_at": older.created_at})
    await store.add_message(older)
    await store.add_message(_comment(older, developer, "a comment"))
    await store.add_message(newer)

    assert await store.list_posts() == [newer, older]


async def test_foreign_keys_are_enforced(store: Store) -> None:
    """An unknown author is a bug, so the database error goes up unchanged."""
    with pytest.raises(IntegrityError):
        await store.add_message(Message(author_id="nobody", text="Hello"))


async def test_times_come_back_in_utc(store: Store) -> None:
    written = datetime(2026, 11, 11, 20, 30, tzinfo=timezone(timedelta(hours=2)))
    user = User(username="me", kind=UserKind.REAL, created_at=written)
    await store.add_user(user)

    found = await store.get_user(user.id)

    assert found is not None
    assert found.created_at == written
    assert found.created_at.tzinfo is UTC
    assert found.created_at.hour == 18


async def test_a_time_without_a_zone_is_refused(store: Store) -> None:
    user = User(username="me", kind=UserKind.REAL, created_at=datetime(2026, 11, 11))

    with pytest.raises(StatementError, match="datetime without a time zone"):
        await store.add_user(user)


async def _post_with_notifications(store: Store) -> tuple[User, User, list[Notification]]:
    me, developer = await _users(store)
    post = Message(author_id=me.id, text="Is Rust growing?")
    comment = _comment(post, developer, "Yes.")
    await store.add_message(post)
    await store.add_message(comment)
    same_moment = datetime(2026, 11, 11, tzinfo=UTC)
    notifications = [
        Notification(user_id=developer.id, post_id=post.id, message_id=post.id, reason=NotificationReason.NEW_POST),
        Notification(user_id=me.id, post_id=post.id, message_id=comment.id, reason=NotificationReason.TOP_LEVEL),
        Notification(user_id=me.id, post_id=post.id, message_id=post.id, reason=NotificationReason.MENTION),
    ]
    notifications = [n.model_copy(update={"created_at": same_moment}) for n in notifications]
    await store.add_notifications(notifications)
    return me, developer, notifications


async def test_notifications_are_listed_per_user_oldest_first(store: Store) -> None:
    me, developer, notifications = await _post_with_notifications(store)

    assert await store.list_notifications(me.id) == notifications[1:]
    assert await store.list_notifications(developer.id) == notifications[:1]


async def test_closed_notifications_keep_their_first_outcome(store: Store) -> None:
    me, _, notifications = await _post_with_notifications(store)
    top_level, mention = notifications[1:]

    await store.close_notifications([top_level.id], NotificationOutcome.SEEN)
    [closed, still_pending] = await store.list_notifications(me.id, include_closed=True)
    await store.close_notifications([top_level.id, "unknown"], NotificationOutcome.SEEN)
    [closed_again, _] = await store.list_notifications(me.id, include_closed=True)

    assert await store.list_notifications(me.id) == [mention]
    assert still_pending == mention
    assert closed.id == top_level.id
    assert closed.outcome is NotificationOutcome.SEEN
    assert closed.closed_at is not None
    assert closed.closed_at.tzinfo is UTC
    assert closed_again == closed


async def test_a_failed_transaction_saves_nothing(store: Store) -> None:
    with pytest.raises(RuntimeError):
        async with store.transaction() as tx:
            await tx.add_user(User(username="me", kind=UserKind.REAL))
            raise RuntimeError

    assert await store.list_users() == []


async def _add_next_user(store: Store) -> None:
    """Reads, waits, then writes on what it read: two of these at once only work one after the other."""
    async with store.transaction() as tx:
        count = len(await tx.list_users())
        await asyncio.sleep(0.05)
        await tx.add_user(User(username=f"user{count}", kind=UserKind.REAL))


async def test_transactions_run_one_after_the_other(store: Store) -> None:
    await asyncio.gather(_add_next_user(store), _add_next_user(store))

    assert [user.username for user in await store.list_users()] == ["user0", "user1"]


async def test_transactions_run_one_after_the_other_across_two_stores(db_path: Path) -> None:
    """Two stores on one file stand in for two processes."""
    async with open_store(_settings(db_path)) as first, open_store(_settings(db_path)) as second:
        await asyncio.gather(_add_next_user(first), _add_next_user(second))

        assert [user.username for user in await first.list_users()] == ["user0", "user1"]


def _revision(db_path: Path) -> str:
    with sqlite3.connect(db_path) as connection:
        return connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]


async def test_a_new_database_is_created_and_logged(db_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="techdebate"):
        async with open_store(_settings(db_path)):
            pass

    assert _revision(db_path) == "0001"
    assert f"created database path={db_path}" in caplog.text


async def test_opening_again_keeps_the_data_and_logs_nothing(db_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    user = User(username="me", kind=UserKind.REAL)
    async with open_store(_settings(db_path)) as store:
        await store.add_user(user)

    with caplog.at_level(logging.INFO, logger="techdebate"):
        async with open_store(_settings(db_path)) as store:
            assert await store.list_users() == [user]

    assert _revision(db_path) == "0001"
    assert "created database" not in caplog.text


async def test_a_database_newer_than_the_code_is_refused(db_path: Path) -> None:
    async with open_store(_settings(db_path)):
        pass
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE alembic_version SET version_num = '9999'")

    with pytest.raises(StoreError, match="revision 9999, which this version of techdebate does not know"):
        async with open_store(_settings(db_path)):
            pass


async def test_the_migrations_build_the_tables_as_defined(db_path: Path) -> None:
    async with open_store(_settings(db_path)):
        pass

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    engine.dispose()

    assert differences == []
