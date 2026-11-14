import re

import pytest

from techdebate.config import Settings
from techdebate.models import Message, User, UserKind
from techdebate.tree import Tree
from techdebate.views import _cut, overview_for, thread_for  # pyright: ignore[reportPrivateUsage]

ME = User(username="me", kind=UserKind.REAL)
DEVELOPER = User(username="developer", kind=UserKind.AGENT)
SCIENTIST = User(username="scientist", kind=UserKind.AGENT)
USERS = {user.id: user for user in (ME, DEVELOPER, SCIENTIST)}

LONG = "word " * 100  # 500 characters, before the stripping that Forum does
LIMIT = 300


def _reply(parent: Message, author: User, text: str) -> Message:
    return Message(post_id=parent.post_id or parent.id, parent_id=parent.id, author_id=author.id, text=text.strip())


def _discussion() -> Tree:
    """#1 and #3 answer the post; #2 answers #1, #4 answers #3, and #5 answers #2, three levels deep."""
    post = Message(author_id=ME.id, text="Are vision transformers replacing CNNs?")
    one = _reply(post, DEVELOPER, f"one {LONG}")
    two = _reply(one, SCIENTIST, f"two {LONG}")
    three = _reply(post, SCIENTIST, "three")
    four = _reply(three, ME, "four")
    five = _reply(two, ME, f"five {LONG}")
    return Tree(post, [one, two, three, four, five])


def _overview(tree: Tree, in_full: list[Message] | None = None) -> str:
    return overview_for(tree, USERS, reply_chars=LIMIT, in_full=in_full or [])


def test_the_overview_shows_the_post_then_every_thread_with_replies_naming_their_parent() -> None:
    tree = Tree(Message(author_id=ME.id, text="Is Rust growing?"), [])
    post = tree.post
    one = _reply(post, DEVELOPER, "Yes.\n\nSee [rust](https://github.com/rust-lang/rust).")
    two = _reply(one, ME, "Why?")
    three = _reply(two, SCIENTIST, "Papers too.")
    four = _reply(post, SCIENTIST, "Mine.")
    tree = Tree(post, [one, two, three, four])

    assert _overview(tree) == (
        "Post by me:\n"
        "Is Rust growing?\n"
        "\n"
        "  #1 Developer [AI]:\n"
        "  Yes.\n"
        "\n"
        "  See [rust](https://github.com/rust-lang/rust).\n"
        "\n"
        "    #2 me, replying to #1:\n"
        "    Why?\n"
        "\n"
        "    #3 Scientist [AI], replying to #2:\n"
        "    Papers too.\n"
        "\n"
        "  #4 Scientist [AI]:\n"
        "  Mine."
    )


def test_a_post_with_no_comments_says_so() -> None:
    tree = Tree(Message(author_id=ME.id, text="Is Rust growing?"), [])

    assert _overview(tree) == "Post by me:\nIs Rust growing?\n\n  No comments yet."


def test_replies_are_cut_and_the_post_and_top_level_comments_are_not() -> None:
    tree = Tree(Message(author_id=ME.id, text=f"post {LONG}".strip()), [])
    top = _reply(tree.post, DEVELOPER, f"top {LONG}")
    reply = _reply(top, SCIENTIST, f"reply {LONG}")
    overview = _overview(Tree(tree.post, [top, reply]))

    assert tree.post.text in overview
    assert top.text in overview
    assert reply.text not in overview
    assert _cut(reply.text, LIMIT) in overview


def test_a_comment_in_full_is_not_cut() -> None:
    tree = _discussion()
    two, five = tree.comment(2), tree.comment(5)
    assert two is not None and five is not None
    overview = _overview(tree, in_full=[five])

    assert five.text in overview
    assert two.text not in overview


def test_the_numbers_agree_between_the_two_views() -> None:
    tree = _discussion()
    overview_headers = re.findall(r"^ *(#\d+ .*):$", _overview(tree), flags=re.MULTILINE)
    thread_headers = [
        header
        for top in (tree.comment(1), tree.comment(3))
        if top is not None
        for header in re.findall(r"^ *(#\d+ .*):$", thread_for(tree, USERS, top), flags=re.MULTILINE)
    ]

    assert sorted(overview_headers) == sorted(thread_headers)
    assert overview_headers == [
        "#1 Developer [AI]",
        "#2 Scientist [AI], replying to #1",
        "#5 me, replying to #2",
        "#3 Scientist [AI]",
        "#4 me, replying to #3",
    ]


def test_a_deep_reply_stays_at_the_second_level() -> None:
    tree = _discussion()

    assert "\n    #5 me, replying to #2:\n" in _overview(tree)


def test_a_thread_holds_only_its_own_comments_in_full() -> None:
    tree = _discussion()
    one, two, three, four, five = (tree.comment(n) for n in range(1, 6))
    assert one is not None and two is not None and three is not None and four is not None and five is not None
    thread = thread_for(tree, USERS, one)

    assert thread.startswith("Post by me:\nAre vision transformers replacing CNNs?\n\n")
    for comment in (one, two, five):
        assert comment.text in thread
    assert "#3" not in thread and "#4" not in thread
    assert three.text not in thread


def test_a_thread_is_the_same_from_any_of_its_comments() -> None:
    tree = _discussion()
    one, two, five = (tree.comment(n) for n in (1, 2, 5))
    assert one is not None and two is not None and five is not None

    assert thread_for(tree, USERS, two) == thread_for(tree, USERS, one)
    assert thread_for(tree, USERS, five) == thread_for(tree, USERS, one)


def test_a_comment_from_another_post_has_no_thread() -> None:
    tree = _discussion()
    other = _reply(Message(author_id=ME.id, text="Other post"), ME, "elsewhere")

    with pytest.raises(ValueError, match="is not on post"):
        thread_for(tree, USERS, other)


def test_a_reply_at_the_limit_is_shown_as_written() -> None:
    text = "a" * (LIMIT - 1) + "b"

    assert _cut(text, LIMIT) == text


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        # The limit falls inside a word: the cut moves back to the whitespace before it.
        ("aaaa bbbb cccc", 7, "aaaa…"),
        # The limit falls just before whitespace: the whole head is kept.
        ("aaaa bbbb cccc", 9, "aaaa bbbb…"),
        # No whitespace before the limit, such as a long URL: an exact cut.
        ("https://example.com/a/very/long/path", 9, "https://e…"),
        # A line break counts as whitespace, and the whitespace before the cut is dropped.
        ("aaaa\n\nbbbb cccc", 7, "aaaa…"),
    ],
)
def test_a_long_reply_is_cut_at_a_word_boundary(text: str, limit: int, expected: str) -> None:
    assert _cut(text, limit) == expected


def test_overview_reply_chars_defaults_to_300() -> None:
    assert Settings(_env_file=None).overview_reply_chars == 300  # pyright: ignore[reportCallIssue]
