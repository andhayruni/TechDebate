import pytest

from techdebate.forum import Forum
from techdebate.models import Message, User, UserKind
from techdebate.store import Store
from techdebate.tree import Tree

ME = "me-id"


def _post(text: str = "Is Rust growing?") -> Message:
    return Message(author_id=ME, text=text)


def _reply(parent: Message, text: str) -> Message:
    return Message(post_id=parent.post_id or parent.id, parent_id=parent.id, author_id=ME, text=text)


def _discussion() -> tuple[Message, list[Message]]:
    """#1 and #3 answer the post; #2 answers #1, #4 answers #3, and #5 answers #2, three levels deep."""
    post = _post()
    one = _reply(post, "one")
    two = _reply(one, "two")
    three = _reply(post, "three")
    four = _reply(three, "four")
    five = _reply(two, "five")
    return post, [one, two, three, four, five]


def test_comments_are_numbered_from_1_in_the_order_they_were_saved() -> None:
    post, comments = _discussion()
    tree = Tree(post, comments)

    assert [tree.number(comment) for comment in comments] == [1, 2, 3, 4, 5]
    assert [tree.comment(n) for n in range(1, 6)] == comments


@pytest.mark.parametrize("number", [0, -1, 6])
def test_an_unknown_number_has_no_comment(number: int) -> None:
    tree = Tree(*_discussion())

    assert tree.comment(number) is None


def test_a_post_without_comments_has_no_threads() -> None:
    tree = Tree(_post(), [])

    assert tree.threads() == []
    assert tree.comment(1) is None


def test_the_parent_is_the_post_for_a_top_level_comment_and_a_comment_for_a_reply() -> None:
    post, (one, two, three, four, five) = _discussion()
    tree = Tree(post, [one, two, three, four, five])

    assert [tree.parent(c) for c in (one, two, three, four, five)] == [post, one, post, three, two]


def test_a_thread_is_its_top_level_comment_and_every_reply_under_it() -> None:
    post, (one, two, three, four, five) = _discussion()
    tree = Tree(post, [one, two, three, four, five])

    assert [tree.top_level(c) for c in (one, two, three, four, five)] == [one, one, three, three, one]
    assert tree.thread(five) == tree.thread(one) == [one, two, five]
    assert tree.thread(four) == [three, four]
    assert tree.threads() == [[one, two, five], [three, four]]


async def test_the_numbers_stay_the_same_as_comments_are_added(store: Store) -> None:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    await store.add_user(me)
    await store.add_user(developer)
    forum = Forum(store, [developer])
    post = await forum.create_post(me, "Is Rust growing?")
    first = await forum.create_comment(developer, post.id, "Yes.")
    reply = await forum.create_comment(me, first.id, "Source?")

    async def numbers() -> dict[str, int]:
        found = await store.get_post(post.id)
        assert found is not None
        tree = Tree.of(found)
        return {comment.id: tree.number(comment) for comment in tree.comments}

    before = await numbers()
    await forum.create_comment(me, post.id, "A second thread.")
    await forum.create_comment(developer, first.id, "GitHub.")  # a later reply in the first thread
    after = await numbers()

    assert before == {first.id: 1, reply.id: 2}
    assert {key: after[key] for key in before} == before
    assert sorted(after.values()) == [1, 2, 3, 4]


def test_a_comment_from_another_post_is_refused() -> None:
    post, comments = _discussion()
    other = _post("Is Go growing?")
    stray = _reply(other, "stray")

    with pytest.raises(ValueError, match="is not on post"):
        Tree(post, [*comments, stray])
    tree = Tree(post, comments)
    for ask in (tree.number, tree.parent, tree.top_level, tree.thread):
        with pytest.raises(ValueError, match="is not on post"):
            ask(stray)


def test_a_reply_listed_before_its_parent_is_refused() -> None:
    post, (one, two, *_) = _discussion()

    with pytest.raises(ValueError, match="which is not an earlier comment"):
        Tree(post, [two, one])
