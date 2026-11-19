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
from pydantic import HttpUrl
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import IntegrityError, StatementError

from techdebate.config import Settings
from techdebate.errors import StoreError
from techdebate.models import (
    Evidence,
    LLMCall,
    Message,
    Notification,
    NotificationOutcome,
    NotificationReason,
    Research,
    ToolResult,
    User,
    UserKind,
)
from techdebate.store import PostSummary, Store, open_store
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


async def test_list_post_summaries_counts_the_comments_at_any_depth_newest_first(store: Store) -> None:
    me, developer = await _users(store)
    older = Message(author_id=me.id, text="older")
    newer = Message(author_id=me.id, text="newer").model_copy(update={"created_at": older.created_at})
    top = _comment(older, developer, "top-level")
    reply = _comment(top, me, "a reply")
    deeper = _comment(reply, developer, "a reply to the reply")
    for message in (older, top, reply, newer, deeper):
        await store.add_message(message)

    assert await store.list_post_summaries() == [
        PostSummary(post=newer, comments=0),
        PostSummary(post=older, comments=3),
    ]


async def test_list_post_summaries_with_no_posts(store: Store) -> None:
    assert await store.list_post_summaries() == []


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


async def _two_posts(store: Store) -> tuple[User, Message, Message]:
    me, developer = await _users(store)
    post = Message(author_id=me.id, text="Is Rust growing?")
    other = Message(author_id=me.id, text="Is Go growing?")
    await store.add_message(post)
    await store.add_message(other)
    return developer, post, other


def _call(agent: User, post: Message, prompt_id: str) -> LLMCall:
    return LLMCall(
        agent_id=agent.id,
        post_id=post.id,
        prompt_id=prompt_id,
        model="claude-haiku-5-5",
        input_tokens=1234,
        output_tokens=56,
        cost_usd=0.000151,
        duration_s=1.7,
        created_at=datetime(2026, 11, 15, tzinfo=UTC),
    )


async def test_llm_calls_are_listed_per_post_oldest_first(store: Store) -> None:
    developer, post, other = await _two_posts(store)
    # Saved in this order, all at the same moment.
    calls = [_call(developer, post, "research.v1"), _call(developer, other, "research.v1")]
    calls += [_call(developer, post, "decide.v1"), _call(developer, post, "developer.v1")]
    for call in calls:
        await store.add_llm_call(call)

    assert await store.list_llm_calls(post.id) == [calls[0], calls[2], calls[3]]
    assert await store.list_llm_calls(other.id) == [calls[1]]
    assert await store.list_llm_calls("unknown") == []


async def test_spent_is_the_cost_of_the_calls_on_the_post(store: Store) -> None:
    developer, post, other = await _two_posts(store)
    for cost, on in [(0.25, post), (0.01, other), (0.02, post)]:
        await store.add_llm_call(_call(developer, on, "decide.v1").model_copy(update={"cost_usd": cost}))

    assert await store.spent_usd(post_id=post.id) == pytest.approx(0.27)
    assert await store.spent_usd(post_id=other.id) == pytest.approx(0.01)
    assert await store.spent_usd(post_id="unknown") == 0.0


async def test_spent_is_narrowed_by_the_post_the_agent_and_the_time_combined(store: Store) -> None:
    developer, post, other = await _two_posts(store)
    scientist = User(username="scientist", kind=UserKind.AGENT)
    await store.add_user(scientist)
    noon = datetime(2026, 11, 16, 12, tzinfo=UTC)
    for agent, on, cost, at in [
        (developer, post, 0.01, noon - timedelta(seconds=1)),
        (developer, other, 0.02, noon),
        (scientist, post, 0.04, noon),
        (developer, post, 0.08, noon + timedelta(hours=1)),
    ]:
        await store.add_llm_call(_call(agent, on, "decide.v1").model_copy(update={"cost_usd": cost, "created_at": at}))

    assert await store.spent_usd() == pytest.approx(0.15)
    assert await store.spent_usd(agent_id=developer.id) == pytest.approx(0.11)
    assert await store.spent_usd(since=noon) == pytest.approx(0.14)  # from `since` on, itself included
    assert await store.spent_usd(agent_id=developer.id, since=noon) == pytest.approx(0.10)
    assert await store.spent_usd(post_id=post.id, agent_id=developer.id, since=noon) == pytest.approx(0.08)
    assert await store.spent_usd(since=noon.astimezone(timezone(timedelta(hours=5)))) == pytest.approx(0.14)
    with pytest.raises(StatementError, match="without a time zone"):
        await store.spent_usd(since=noon.replace(tzinfo=None))


async def test_an_llm_call_needs_a_known_agent_and_post(store: Store) -> None:
    developer, post, _ = await _two_posts(store)

    with pytest.raises(IntegrityError):
        await store.add_llm_call(_call(developer, post, "decide.v1").model_copy(update={"post_id": "unknown"}))
    with pytest.raises(IntegrityError):
        await store.add_llm_call(_call(developer, post, "decide.v1").model_copy(update={"agent_id": "nobody"}))


def _results() -> list[ToolResult]:
    """One of each outcome: evidence, no information, a failed tool and a tool not used."""
    evidence = Evidence(
        source="github_search",
        query="rust",
        url=HttpUrl("https://github.com/search?q=rust&type=repositories"),
        title="GitHub repositories matching 'rust'",
        summary="12,345 repositories match 'rust' on GitHub",
        value=12345,
        fetched_at=datetime(2026, 11, 15, 9, 30, tzinfo=UTC),
    )
    return [
        ToolResult(tool="github_search", query="rust", evidence=[evidence]),
        ToolResult(tool="github_repo", query="rust-lang/rust"),
        ToolResult(tool="github_repo", query="rust", failed=True),
        ToolResult(tool="pypi", query=None),
    ]


async def test_research_round_trips_with_every_outcome(store: Store) -> None:
    developer, post, _ = await _two_posts(store)
    research = Research(agent_id=developer.id, post_id=post.id, results=_results())

    await store.add_research(research)

    assert await store.get_research(developer.id, post.id) == research


async def test_research_is_none_for_another_agent_or_post(store: Store) -> None:
    developer, post, other = await _two_posts(store)
    me = await store.find_user("me")
    assert me is not None
    await store.add_research(Research(agent_id=developer.id, post_id=post.id, results=_results()))

    assert await store.get_research(developer.id, other.id) is None
    assert await store.get_research(me.id, post.id) is None


async def test_an_agent_researches_a_post_once(store: Store) -> None:
    """A second save is a bug, so the database error goes up unchanged, and the first research stays."""
    developer, post, _ = await _two_posts(store)
    first = Research(agent_id=developer.id, post_id=post.id, results=_results())
    await store.add_research(first)

    with pytest.raises(IntegrityError):
        await store.add_research(Research(agent_id=developer.id, post_id=post.id, results=[]))

    assert await store.get_research(developer.id, post.id) == first


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


_HEAD = "0003"  # the newest migration


def _revision(db_path: Path) -> str:
    with sqlite3.connect(db_path) as connection:
        return connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]


async def test_a_new_database_is_created_and_logged(db_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="techdebate"):
        async with open_store(_settings(db_path)):
            pass

    assert _revision(db_path) == _HEAD
    assert f"created database path={db_path}" in caplog.text


async def test_opening_again_keeps_the_data_and_logs_nothing(db_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    user = User(username="me", kind=UserKind.REAL)
    async with open_store(_settings(db_path)) as store:
        await store.add_user(user)

    with caplog.at_level(logging.INFO, logger="techdebate"):
        async with open_store(_settings(db_path)) as store:
            assert await store.list_users() == [user]

    assert _revision(db_path) == _HEAD
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
