"""What an agent reads about a post: the overview of the whole post in short form, and one thread in full. These are the
only places that build it, from one formatter, so a comment's number and header are the same in both views."""

import re
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence

from techdebate.models import Message, User
from techdebate.tree import Tree
from techdebate.users import shown_name

_CUT_MARK = "…"
# The last run of whitespace in a text, and whatever follows it up to the end.
_LAST_SPACE = re.compile(r"\s+\S*\Z")


def overview_for(tree: Tree, users: Mapping[str, User], *, reply_chars: int, in_full: Iterable[Message] = ()) -> str:
    """The post and every thread. The post and top-level comments are shown in full, and replies are cut to
    `reply_chars`, except the comments in `in_full`. `users` maps each author's ID to its user."""
    full = {comment.id for comment in in_full}

    def text(comment: Message) -> str:
        if comment.parent_id == tree.post.id or comment.id in full:
            return comment.text
        return _cut(comment.text, reply_chars)

    return _format(tree, users, tree.threads(), text)


def thread_for(tree: Tree, users: Mapping[str, User], comment: Message) -> str:
    """The post and the thread of `comment`, which can be any comment in it, all in full."""
    return _format(tree, users, [tree.thread(comment)], lambda message: message.text)


def _format(
    tree: Tree, users: Mapping[str, User], threads: Sequence[list[Message]], text: Callable[[Message], str]
) -> str:
    """The post, then each thread: its top-level comment one level in, and every reply under it a second level in.
    Every reply names the comment it answers, so the parent never has to be read from the indentation."""

    def block(header: str, body: str, level: int) -> str:
        return textwrap.indent(f"{header}:\n{body}", "  " * level)

    def author(message: Message) -> str:
        return shown_name(users[message.author_id])

    blocks = [block(f"Post by {author(tree.post)}", tree.post.text, 0)]
    if not tree.comments:
        blocks.append(textwrap.indent("No comments yet.", "  "))
    for top, *replies in threads:
        blocks.append(block(f"#{tree.number(top)} {author(top)}", text(top), 1))
        for reply in replies:
            header = f"#{tree.number(reply)} {author(reply)}, replying to #{tree.number(tree.parent(reply))}"
            blocks.append(block(header, text(reply), 2))
    return "\n\n".join(blocks)


def _cut(text: str, limit: int) -> str:
    """The text, or its start up to `limit` characters followed by the cut mark. The cut falls at the last whitespace
    before the limit, so no word or URL is split, or exactly at the limit when there is none."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if not text[limit].isspace():  # the limit falls inside a word: drop that word
        last_space = _LAST_SPACE.search(head)
        if last_space is not None and last_space.start() > 0:
            head = head[: last_space.start()]
    return head.rstrip() + _CUT_MARK
