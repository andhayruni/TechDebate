import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import pytest

from techdebate.errors import ForumError, TechDebateError
from techdebate.forum import Forum
from techdebate.models import Message, Notification, NotificationOutcome, NotificationReason, User, UserKind
from techdebate.store import Store
from techdebate.tree import Tree

MENTION = NotificationReason.MENTION


async def _setup(store: Store) -> tuple[Forum, User, User]:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    await store.add_user(me)
    await store.add_user(developer)
    return Forum(store, [developer]), me, developer


async def _pending(store: Store, *users: User) -> list[tuple[str, str, NotificationReason]]:
    """(username, message text, reason) for each pending notification of these users, oldest first per user."""
    found: list[tuple[str, str, NotificationReason]] = []
    for user in users:
        for notification in await store.list_notifications(user.id):
            message = await store.get_message(notification.message_id)
            assert message is not None
            found.append((user.username, message.text, notification.reason))
    return found


async def _close_all(store: Store, user: User) -> None:
    await store.close_notifications([n.id for n in await store.list_notifications(user.id)], NotificationOutcome.SEEN)


async def _saved(store: Store, post: Message) -> list[Message]:
    found = await store.get_post(post.id)
    assert found is not None
    return [found.post, *found.comments]


async def test_a_real_user_creates_a_post(store: Store) -> None:
    forum, me, _ = await _setup(store)

    post = await forum.create_post(me, "Is Rust growing?")

    assert (post.author_id, post.text, post.post_id, post.parent_id) == (me.id, "Is Rust growing?", None, None)
    assert await store.get_message(post.id) == post


async def test_an_agent_cannot_create_a_post_and_nothing_is_saved(store: Store) -> None:
    forum, _, developer = await _setup(store)

    with pytest.raises(ForumError, match="agents cannot create posts, and developer is an agent"):
        await forum.create_post(developer, "Is Rust growing?")
    assert await store.list_posts() == []


async def test_a_refused_write_is_a_one_line_error() -> None:
    assert issubclass(ForumError, TechDebateError)


async def test_real_users_and_agents_comment_and_reply(store: Store) -> None:
    forum, me, developer = await _setup(store)
    post = await forum.create_post(me, "Is Rust growing?")

    comment = await forum.create_comment(developer, post.id, "Yes.")
    reply = await forum.create_comment(me, comment.id, "Source?")
    answer = await forum.create_comment(developer, reply.id, "GitHub.")

    assert [(m.author_id, m.post_id, m.parent_id) for m in (comment, reply, answer)] == [
        (developer.id, post.id, post.id),
        (me.id, post.id, comment.id),
        (developer.id, post.id, reply.id),
    ]
    assert await _saved(store, post) == [post, comment, reply, answer]


async def test_replies_three_and_more_levels_deep_keep_their_post_and_thread(store: Store) -> None:
    forum, me, developer = await _setup(store)
    post = await forum.create_post(me, "Is Rust growing?")
    first = await forum.create_comment(developer, post.id, "Yes.")
    second = await forum.create_comment(me, post.id, "I think so too.")
    chain = [first]
    for depth in range(2, 6):  # replies at depths 2 to 5, each to the one before
        chain.append(await forum.create_comment(me if depth % 2 else developer, chain[-1].id, f"depth {depth}"))
    other = await forum.create_comment(developer, second.id, "Agreed.")

    found = await store.get_post(post.id)
    assert found is not None
    tree = Tree.of(found)
    assert all(reply.post_id == post.id for reply in chain)
    for reply in chain:
        assert tree.top_level(reply) == first
        assert tree.thread(reply) == chain
    assert tree.thread(other) == [second, other]


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
async def test_a_blank_post_is_refused(store: Store, text: str) -> None:
    forum, me, _ = await _setup(store)

    with pytest.raises(ForumError, match="the text is blank"):
        await forum.create_post(me, text)
    assert await store.list_posts() == []


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
async def test_a_blank_comment_is_refused_and_nothing_is_saved(store: Store, text: str) -> None:
    forum, me, developer = await _setup(store)
    post = await forum.create_post(me, "Is Rust growing?")

    with pytest.raises(ForumError, match="the text is blank"):
        await forum.create_comment(developer, post.id, text)
    assert await _saved(store, post) == [post]


async def test_a_comment_on_an_unknown_parent_is_refused_and_nothing_is_saved(store: Store) -> None:
    forum, me, developer = await _setup(store)
    post = await forum.create_post(me, "Is Rust growing?")

    with pytest.raises(ForumError, match="there is no post or comment with the ID 'unknown'"):
        await forum.create_comment(developer, "unknown", "Yes.")
    assert await _saved(store, post) == [post]


async def test_the_text_is_saved_without_its_surrounding_whitespace(store: Store) -> None:
    forum, me, developer = await _setup(store)

    post = await forum.create_post(me, "    Is Rust growing?\n")
    comment = await forum.create_comment(developer, post.id, "\n  Yes.\n\n  See   the repo.  \n")

    assert post.text == "Is Rust growing?"
    assert comment.text == "Yes.\n\n  See   the repo."  # the whitespace inside stays as written
    assert await _saved(store, post) == [post, comment]


# Notifications


async def test_notifications_are_saved_with_their_message(store: Store) -> None:
    forum, me, developer = await _setup(store)

    post = await forum.create_post(me, "Is Rust growing?")
    comment = await forum.create_comment(developer, post.id, "Yes.")

    [for_developer] = await store.list_notifications(developer.id)
    [for_me] = await store.list_notifications(me.id)
    assert (for_developer.post_id, for_developer.message_id, for_developer.reason) == (
        post.id,
        post.id,
        NotificationReason.NEW_POST,
    )
    assert (for_me.post_id, for_me.message_id, for_me.reason) == (post.id, comment.id, NotificationReason.TOP_LEVEL)


async def test_a_refused_write_saves_no_notifications(store: Store) -> None:
    forum, me, developer = await _setup(store)
    post = await forum.create_post(me, "Is Rust growing?")

    for write in (forum.create_comment(me, post.id, " "), forum.create_comment(me, "unknown", "@developer")):
        with pytest.raises(ForumError):
            await write

    assert await _pending(store, me, developer) == [("developer", "Is Rust growing?", NotificationReason.NEW_POST)]


async def test_mentions_are_looked_up_and_names_that_are_not_users_ignored(store: Store) -> None:
    forum, me, developer = await _setup(store)
    ann = User(username="ann", kind=UserKind.REAL)
    await store.add_user(ann)
    post = await forum.create_post(me, "Is Rust growing?")

    await forum.create_comment(me, post.id, "@Developer and @ann, and @nobody or write to me@ann.io")

    assert await _pending(store, ann) == [("ann", "@Developer and @ann, and @nobody or write to me@ann.io", MENTION)]
    assert [n.reason for n in await store.list_notifications(developer.id)] == [NotificationReason.NEW_POST, MENTION]


async def test_a_mention_notifies_once_and_does_not_join_the_thread(store: Store) -> None:
    forum, me, developer = await _setup(store)
    scientist = User(username="scientist", kind=UserKind.AGENT)
    await store.add_user(scientist)
    forum = Forum(store, [developer, scientist])
    post = await forum.create_post(me, "Is Rust growing?")
    first = await forum.create_comment(developer, post.id, "Yes.")
    await _close_all(store, scientist)

    await forum.create_comment(me, first.id, "What do you think, @scientist?")
    await forum.create_comment(developer, first.id, "GitHub says so.")

    assert await _pending(store, scientist) == [("scientist", "What do you think, @scientist?", MENTION)]


async def test_an_agent_outside_the_registry_is_not_notified(store: Store) -> None:
    forum, me, developer = await _setup(store)
    retired = User(username="retired", kind=UserKind.AGENT)
    await store.add_user(retired)
    with_retired = Forum(store, [developer, retired])  # before its role was removed
    post = await with_retired.create_post(me, "Is Rust growing?")
    first = await with_retired.create_comment(retired, post.id, "Yes.")
    await _close_all(store, retired)

    await forum.create_comment(me, first.id, "Thanks, @retired.")
    await forum.create_post(me, "Is Go growing?")

    assert await store.list_notifications(retired.id) == []


# Listeners


async def test_listeners_hear_of_each_saved_message_and_its_notifications_after_the_commit(
    store: Store, db_path: Path
) -> None:
    forum, me, developer = await _setup(store)
    heard: list[tuple[str, list[str], bool]] = []

    def listener(message: Message, notifications: Sequence[Notification]) -> None:
        # A connection of its own sees the message only once its transaction has committed.
        with closing(sqlite3.connect(db_path)) as db:
            committed = db.execute("SELECT 1 FROM messages WHERE id = ?", (message.id,)).fetchone() is not None
        heard.append((message.text, [n.user_id for n in notifications], committed))

    forum.subscribe(listener)

    post = await forum.create_post(me, "Is Rust growing?")
    await forum.create_comment(developer, post.id, "Yes.")

    assert heard == [("Is Rust growing?", [developer.id], True), ("Yes.", [me.id], True)]


async def test_a_listener_that_raises_is_logged_and_the_others_still_hear(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    forum, me, _ = await _setup(store)
    heard: list[str] = []

    def broken(message: Message, notifications: Sequence[Notification]) -> None:
        raise RuntimeError("broken listener")

    forum.subscribe(broken)
    forum.subscribe(lambda message, notifications: heard.append(message.text))

    post = await forum.create_post(me, "Is Rust growing?")

    assert heard == ["Is Rust growing?"]
    assert await store.get_message(post.id) == post
    assert "forum listener failed" in caplog.text and "broken listener" in caplog.text


async def test_a_refused_write_tells_no_listener(store: Store) -> None:
    forum, me, _ = await _setup(store)
    heard: list[str] = []
    forum.subscribe(lambda message, notifications: heard.append(message.text))

    with pytest.raises(ForumError):
        await forum.create_post(me, "   ")

    assert heard == []
