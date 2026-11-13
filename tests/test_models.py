from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from techdebate.config import Settings
from techdebate.models import Message, Notification, NotificationOutcome, NotificationReason, User, UserKind


def test_a_post_has_no_parent_and_no_post() -> None:
    me = User(username="me", kind=UserKind.REAL)

    post = Message(author_id=me.id, text="Is Rust growing?")

    assert (post.post_id, post.parent_id) == (None, None)


def test_a_reply_points_to_its_parent_and_to_the_post_at_the_root() -> None:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    post = Message(author_id=me.id, text="Is Rust growing?")

    comment = Message(post_id=post.id, parent_id=post.id, author_id=developer.id, text="Yes.")
    reply = Message(post_id=post.id, parent_id=comment.id, author_id=me.id, text="Why?")

    assert (comment.post_id, comment.parent_id) == (post.id, post.id)
    assert (reply.post_id, reply.parent_id) == (post.id, comment.id)


@pytest.mark.parametrize(("post_id", "parent_id"), [("p", None), (None, "p")])
def test_a_message_needs_both_post_and_parent_or_neither(post_id: str | None, parent_id: str | None) -> None:
    with pytest.raises(ValidationError, match="a post has neither post_id nor parent_id"):
        Message(post_id=post_id, parent_id=parent_id, author_id="a", text="x")


def test_a_notification_is_pending_until_closed_with_an_outcome() -> None:
    pending = Notification(user_id="u", post_id="p", message_id="p", reason=NotificationReason.NEW_POST)
    closed = pending.model_copy(update={"closed_at": datetime.now(UTC), "outcome": NotificationOutcome.SEEN})

    assert pending.pending
    assert not closed.pending


@pytest.mark.parametrize(("closed_at", "outcome"), [(datetime.now(UTC), None), (None, NotificationOutcome.SEEN)])
def test_a_notification_needs_both_closed_at_and_outcome_or_neither(
    closed_at: datetime | None, outcome: NotificationOutcome | None
) -> None:
    with pytest.raises(ValidationError, match="both closed_at and outcome, or neither"):
        Notification(
            user_id="u",
            post_id="p",
            message_id="p",
            reason=NotificationReason.NEW_POST,
            closed_at=closed_at,
            outcome=outcome,
        )


def test_users_and_messages_round_trip() -> None:
    user = User(username="developer", kind=UserKind.AGENT)
    message = Message(post_id="p", parent_id="p", author_id=user.id, text="Yes.")

    assert User.model_validate_json(user.model_dump_json()) == user
    assert Message.model_validate_json(message.model_dump_json()) == message


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert settings.username == "me"
    assert settings.post_budget_usd <= settings.daily_budget_usd
