"""The forum: the one way to write a post or a comment, so a message is treated the same whoever wrote it. It checks
what the author may do and the message itself, works out who is notified, and saves the message with its notifications
in one transaction."""

from collections.abc import Iterable

from techdebate.errors import ForumError
from techdebate.models import Message, User, UserKind
from techdebate.notify import find_mentions, notifications_for
from techdebate.store import Store, StoreOps
from techdebate.tree import Tree


class Forum:
    """Real users can post and comment, and agents can only comment. Reads go through the Store."""

    def __init__(self, store: Store, agents: Iterable[User]) -> None:
        """`agents` are the agents in the role registry, from the sync: only they are notified."""
        self._store = store
        self._agents = tuple(agents)

    async def create_post(self, author: User, text: str) -> Message:
        """The saved post. Raises ForumError for an agent or for blank text."""
        # Trusted: the author was read from the database, by the sync or (later) by login.
        if author.kind is UserKind.AGENT:
            raise ForumError(f"agents cannot create posts, and {author.username} is an agent")
        post = Message(author_id=author.id, text=_checked(text))
        async with self._store.transaction() as tx:
            await self._save(tx, post, author, Tree(post, []))
        return post

    async def create_comment(self, author: User, parent_id: str, text: str) -> Message:
        """The saved comment, under the post or the comment with parent_id. Raises ForumError for blank text or an
        unknown parent."""
        text = _checked(text)
        async with self._store.transaction() as tx:
            parent = await tx.get_message(parent_id)
            if parent is None:
                raise ForumError(f"there is no post or comment with the ID {parent_id!r}")
            post_id = parent.post_id or parent.id
            comment = Message(post_id=post_id, parent_id=parent.id, author_id=author.id, text=text)
            found = await tx.get_post(post_id)
            assert found is not None, "a saved parent always has its post"
            await self._save(tx, comment, author, Tree(found.post, [*found.comments, comment]))
        return comment

    async def _save(self, tx: StoreOps, message: Message, author: User, tree: Tree) -> None:
        """Works out who is notified from what tx reads, and saves the message with its notifications. Everything is
        read in the write's transaction, so two writers at the same time cannot miss each other's messages."""
        mentioned: list[User] = []
        if author.kind is UserKind.REAL:  # an agent's @name is plain text, so there is nothing to look up
            for name in find_mentions(message.text):
                if (user := await tx.find_user(name)) is not None:
                    mentioned.append(user)
        notifications = notifications_for(
            message,
            tree,
            author=author,
            mentioned=mentioned,
            agents=self._agents,
            all_agents=await tx.list_users(UserKind.AGENT),
        )
        await tx.add_message(message)
        await tx.add_notifications(notifications)


def _checked(text: str) -> str:
    """The text without its surrounding whitespace, so a leading indent never becomes a markdown code block. Raises
    ForumError when nothing is left."""
    text = text.strip()
    if not text:
        raise ForumError("the text is blank")
    return text
