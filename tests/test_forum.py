import pytest

from techdebate.errors import ForumError, TechDebateError
from techdebate.forum import Forum
from techdebate.models import Message, User, UserKind
from techdebate.store import Store
from techdebate.tree import Tree


async def _setup(store: Store) -> tuple[Forum, User, User]:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    await store.add_user(me)
    await store.add_user(developer)
    return Forum(store), me, developer


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
