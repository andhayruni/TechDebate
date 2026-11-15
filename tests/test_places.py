import pytest

from techdebate.models import Message, ReplyTo
from techdebate.places import POST, Places, one_per_thread, place_message, places_for
from techdebate.tree import Tree

ME = "me-id"
AGENT = "agent-id"
OTHER = "other-id"


class _Discussion:
    """Builds a post's comments in order, so comment #n is the n-th one added."""

    def __init__(self) -> None:
        self.post = Message(author_id=ME, text="Is Rust growing?")
        self.comments: list[Message] = []

    def add(self, author: str, parent: int = POST) -> "_Discussion":
        answers = self.post if parent == POST else self.comments[parent - 1]
        self.comments.append(Message(post_id=self.post.id, parent_id=answers.id, author_id=author, text="text"))
        return self

    def tree(self) -> Tree:
        return Tree(self.post, self.comments)


def _places(discussion: _Discussion, max_per_thread: int = 3) -> Places:
    return places_for(discussion.tree(), AGENT, max_per_thread=max_per_thread)


# places_for


def test_a_post_with_no_comments_offers_only_the_post() -> None:
    assert _places(_Discussion()) == Places(open=(POST,), has_top_level=False, full_threads=(), limit=3)


def test_every_comment_by_someone_else_is_open_in_a_thread_below_the_limit() -> None:
    # #1 me, #2 other replying to #1, #3 other on the post, #4 me replying to #2, three levels deep
    discussion = _Discussion().add(ME).add(OTHER, 1).add(OTHER).add(ME, 2)

    assert _places(discussion).open == (POST, 1, 2, 3, 4)


def test_the_post_is_closed_once_the_agent_has_a_top_level_comment() -> None:
    discussion = _Discussion().add(AGENT).add(ME, 1)

    places = _places(discussion)

    assert places.open == (2,)
    assert places.has_top_level


def test_a_reply_does_not_use_the_top_level_comment() -> None:
    discussion = _Discussion().add(ME).add(AGENT, 1)

    places = _places(discussion)

    assert places.open == (POST, 1)
    assert not places.has_top_level


def test_the_agent_never_replies_to_itself() -> None:
    discussion = _Discussion().add(ME).add(AGENT, 1).add(ME, 2).add(AGENT, 3)

    assert _places(discussion).open == (POST, 1, 3)


def test_a_thread_at_the_limit_is_closed_and_named() -> None:
    # Thread #1: me, then the agent's replies #2 and #4, with my #3 between. Thread #5: me.
    discussion = _Discussion().add(ME).add(AGENT, 1).add(ME, 2).add(AGENT, 3).add(ME)

    places = _places(discussion, max_per_thread=2)

    assert places.open == (POST, 5)
    assert places.full_threads == (1,)
    assert places.limit == 2


def test_the_count_covers_the_top_level_comment_and_replies_to_anyone() -> None:
    # The agent's top-level #1, my reply #2, the agent's reply to me #3, another agent's #4, the agent's reply #5.
    discussion = _Discussion().add(AGENT).add(ME, 1).add(AGENT, 2).add(OTHER, 3).add(AGENT, 4)

    places = _places(discussion)

    assert places == Places(open=(), has_top_level=True, full_threads=(1,), limit=3)


def test_one_comment_below_the_limit_keeps_the_thread_open() -> None:
    discussion = _Discussion().add(AGENT).add(ME, 1).add(AGENT, 2).add(OTHER, 3)

    assert _places(discussion).open == (2, 4)


def test_each_thread_is_counted_on_its_own() -> None:
    discussion = _Discussion().add(ME).add(AGENT, 1).add(OTHER).add(AGENT, 3)

    places = _places(discussion, max_per_thread=1)

    assert places.open == (POST,)
    assert places.full_threads == (1, 3)


def test_no_open_place_at_all() -> None:
    discussion = _Discussion().add(AGENT)

    assert _places(discussion, max_per_thread=1).open == ()


# one_per_thread


def _picks(*places: int) -> list[ReplyTo]:
    return [ReplyTo(place=place, point=f"point {place}") for place in places]


def test_two_places_in_one_thread_keep_the_first_named() -> None:
    # Thread #1 holds #1, #2 and #4. Thread #3 holds #3.
    tree = _Discussion().add(ME).add(OTHER, 1).add(OTHER).add(ME, 2).tree()

    assert one_per_thread(tree, _picks(4, 3, 1, 2)) == _picks(4, 3)


def test_the_post_is_a_place_of_its_own() -> None:
    tree = _Discussion().add(ME).add(OTHER, 1).tree()

    assert one_per_thread(tree, _picks(2, POST, 1)) == _picks(2, POST)


def test_the_post_named_twice_is_kept_once() -> None:
    tree = _Discussion().tree()

    assert one_per_thread(tree, _picks(POST, POST)) == _picks(POST)


def test_a_pass_stays_a_pass() -> None:
    assert one_per_thread(_Discussion().tree(), []) == []


# place_message


def test_a_place_maps_to_the_post_or_the_comment() -> None:
    discussion = _Discussion().add(ME).add(OTHER, 1)
    tree = discussion.tree()

    assert place_message(tree, POST) == discussion.post
    assert place_message(tree, 2) == discussion.comments[1]


@pytest.mark.parametrize("place", [-1, 3])
def test_an_unknown_place_is_a_bug(place: int) -> None:
    tree = _Discussion().add(ME).add(OTHER, 1).tree()

    with pytest.raises(ValueError, match=f"no comment #{place}"):
        place_message(tree, place)
