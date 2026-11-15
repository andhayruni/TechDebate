"""Where an agent may comment on a post. A place is a number: 0 is the post itself, for a top-level comment, and n is
comment #n, for a reply. Each agent gets one top-level comment per post, a limited number of comments in each thread,
and never replies to itself."""

from collections.abc import Sequence
from dataclasses import dataclass

from techdebate.models import Message, ReplyTo
from techdebate.tree import Tree

POST = 0  # the place for a top-level comment: the post itself


@dataclass(frozen=True)
class Places:
    """The places open to one agent on one post, and why the others are closed."""

    open: tuple[int, ...]  # 0 first when it is open, then the comment numbers in order
    has_top_level: bool  # the agent has used its top-level comment, so 0 is closed
    full_threads: tuple[int, ...]  # the threads it has filled, by the number of their top-level comment
    limit: int  # the comments an agent may write in one thread


def places_for(tree: Tree, agent_id: str, *, max_per_thread: int) -> Places:
    """Every comment that is not the agent's own, in a thread where it has written fewer than `max_per_thread`
    comments, and the post while the agent has no top-level comment on it."""
    has_top_level = any(c.parent_id == tree.post.id and c.author_id == agent_id for c in tree.comments)
    open_places = [] if has_top_level else [POST]
    full: list[int] = []
    for thread in tree.threads():
        if sum(c.author_id == agent_id for c in thread) >= max_per_thread:
            full.append(tree.number(thread[0]))
        else:
            open_places.extend(tree.number(c) for c in thread if c.author_id != agent_id)
    return Places(
        open=tuple(sorted(open_places)), has_top_level=has_top_level, full_threads=tuple(full), limit=max_per_thread
    )


def one_per_thread(tree: Tree, reply_to: Sequence[ReplyTo]) -> list[ReplyTo]:
    """The places with one comment per thread, keeping the first named in each. The post is a place of its own."""
    kept: dict[str, ReplyTo] = {}
    for item in reply_to:
        message = place_message(tree, item.place)
        key = message.id if item.place == POST else tree.top_level(message).id
        kept.setdefault(key, item)
    return list(kept.values())


def place_message(tree: Tree, place: int) -> Message:
    """The message a comment at `place` answers: the post for 0, or comment #place."""
    if place == POST:
        return tree.post
    comment = tree.comment(place)
    if comment is None:
        raise ValueError(f"there is no comment #{place} on post {tree.post.id}")
    return comment
