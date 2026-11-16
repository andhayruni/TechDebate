"""The Store: the one way the rest of the code reads and writes the database. It takes and returns the records in
models.py, and nothing from the database library leaves it."""

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from techdebate.models import LLMCall, Message, Notification, NotificationOutcome, Research, User, UserKind


@dataclass(frozen=True)
class PostWithComments:
    post: Message
    comments: list[Message]  # every comment under the post, at any depth, oldest first


class StoreOps(Protocol):
    """What can be read and written. Lists come back oldest first unless they say otherwise."""

    async def add_user(self, user: User) -> None:
        """Raises StoreError when the username is taken, by a user of either kind."""
        ...

    async def get_user(self, user_id: str) -> User | None: ...

    async def find_user(self, username: str) -> User | None: ...

    async def list_users(self, kind: UserKind | None = None) -> list[User]:
        """By username."""
        ...

    async def add_message(self, message: Message) -> None: ...

    async def get_message(self, message_id: str) -> Message | None: ...

    async def get_post(self, post_id: str) -> PostWithComments | None:
        """The post and all its comments, in one lookup. None when there is no post with that ID."""
        ...

    async def list_posts(self) -> list[Message]:
        """Newest first."""
        ...

    async def add_notifications(self, notifications: Sequence[Notification]) -> None: ...

    async def list_notifications(self, user_id: str, *, include_closed: bool = False) -> list[Notification]:
        """The user's pending notifications, or all of them with include_closed."""
        ...

    async def close_notifications(self, notification_ids: Sequence[str], outcome: NotificationOutcome) -> None:
        """Closes the ones still pending, now. A closed notification keeps its first outcome."""
        ...

    async def add_llm_call(self, call: LLMCall) -> None: ...

    async def list_llm_calls(self, post_id: str) -> list[LLMCall]:
        """Every AI call made on the post, by any agent."""
        ...

    async def spent_usd(
        self, *, post_id: str | None = None, agent_id: str | None = None, since: datetime | None = None
    ) -> float:
        """The cost of the stored AI calls, 0 with none: on the post, by the agent and from `since` on, for each filter
        given, and of every call without one."""
        ...

    async def add_research(self, research: Research) -> None:
        """An agent researches a post once: a second save for the same agent and post is a bug, and raises."""
        ...

    async def get_research(self, agent_id: str, post_id: str) -> Research | None:
        """None when the agent has not researched the post."""
        ...


class Store(StoreOps, Protocol):
    """Each call runs in a short transaction of its own. `transaction()` gives the same calls in one transaction that
    no other writer, in this process or another, can come between: a writer that reads and then writes on what it
    read uses it. No AI call is made inside one."""

    def transaction(self) -> AbstractAsyncContextManager[StoreOps]: ...
