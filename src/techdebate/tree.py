"""The shape of one post's comments: their numbers, the message a comment answers, and the threads. A thread is a
top-level comment (one that answers the post) and every reply under it, at any depth."""

from collections.abc import Sequence
from typing import Self

from techdebate.models import Message
from techdebate.store import PostWithComments


class Tree:
    """Built once from a post and all its comments, oldest first. Comment #n is the n-th comment in that order, starting
    at 1, so its number never changes once it is given. The post itself has no number."""

    def __init__(self, post: Message, comments: Sequence[Message]) -> None:
        self.post = post
        self.comments: tuple[Message, ...] = tuple(comments)
        self._numbers: dict[str, int] = {}
        self._top_levels: dict[str, Message] = {}  # by comment ID
        for number, comment in enumerate(self.comments, start=1):
            if comment.post_id != post.id:
                raise ValueError(f"comment {comment.id} is not on post {post.id}")
            if comment.parent_id == post.id:
                self._top_levels[comment.id] = comment
            elif comment.parent_id in self._top_levels:  # a parent is always saved before its replies
                self._top_levels[comment.id] = self._top_levels[comment.parent_id]
            else:
                raise ValueError(f"comment {comment.id} answers {comment.parent_id}, which is not an earlier comment")
            self._numbers[comment.id] = number

    @classmethod
    def of(cls, found: PostWithComments) -> Self:
        """The tree of what store.get_post returns."""
        return cls(found.post, found.comments)

    def number(self, comment: Message) -> int:
        return self._numbers[self._known(comment)]

    def comment(self, number: int) -> Message | None:
        """Comment #number, or None when the post has no comment with that number."""
        return self.comments[number - 1] if 1 <= number <= len(self.comments) else None

    def parent(self, comment: Message) -> Message:
        """The message the comment answers: the post for a top-level comment, else a comment."""
        self._known(comment)
        if comment.parent_id == self.post.id:
            return self.post
        return self.comments[self._numbers[str(comment.parent_id)] - 1]

    def top_level(self, comment: Message) -> Message:
        """The top-level comment of the comment's thread, which is the comment itself when it answers the post."""
        return self._top_levels[self._known(comment)]

    def thread(self, comment: Message) -> list[Message]:
        """The comment's whole thread: its top-level comment and every reply under it, oldest first."""
        top = self.top_level(comment)
        return [other for other in self.comments if self._top_levels[other.id].id == top.id]

    def threads(self) -> list[list[Message]]:
        """Every thread on the post, each oldest first, in the order their top-level comments were saved."""
        threads: dict[str, list[Message]] = {}
        for comment in self.comments:
            threads.setdefault(self._top_levels[comment.id].id, []).append(comment)
        return list(threads.values())

    def _known(self, comment: Message) -> str:
        if comment.id not in self._numbers:
            raise ValueError(f"comment {comment.id} is not on post {self.post.id}")
        return comment.id
