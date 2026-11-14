"""The forum: the one way to write a post or a comment, so a message is treated the same whoever wrote it. It checks
what the author may do and the message itself, and saves it in one transaction."""

from techdebate.errors import ForumError
from techdebate.models import Message, User, UserKind
from techdebate.store import Store


class Forum:
    """Real users can post and comment, and agents can only comment. Reads go through the Store."""

    def __init__(self, store: Store) -> None:
        self._store = store

    async def create_post(self, author: User, text: str) -> Message:
        """The saved post. Raises ForumError for an agent or for blank text."""
        # Trusted: the author was read from the database, by the sync or (later) by login.
        if author.kind is UserKind.AGENT:
            raise ForumError(f"agents cannot create posts, and {author.username} is an agent")
        post = Message(author_id=author.id, text=_checked(text))
        async with self._store.transaction() as tx:
            await tx.add_message(post)
        return post

    async def create_comment(self, author: User, parent_id: str, text: str) -> Message:
        """The saved comment, under the post or the comment with parent_id. Raises ForumError for blank text or an
        unknown parent."""
        text = _checked(text)
        async with self._store.transaction() as tx:
            parent = await tx.get_message(parent_id)
            if parent is None:
                raise ForumError(f"there is no post or comment with the ID {parent_id!r}")
            comment = Message(post_id=parent.post_id or parent.id, parent_id=parent.id, author_id=author.id, text=text)
            await tx.add_message(comment)
        return comment


def _checked(text: str) -> str:
    """The text without its surrounding whitespace, so a leading indent never becomes a markdown code block. Raises
    ForumError when nothing is left."""
    text = text.strip()
    if not text:
        raise ForumError("the text is blank")
    return text
