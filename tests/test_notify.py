import pytest

from techdebate.models import Message, NotificationReason, User, UserKind
from techdebate.notify import find_mentions, notifications_for
from techdebate.tree import Tree

ME = User(username="me", kind=UserKind.REAL)
ANN = User(username="ann", kind=UserKind.REAL)
DEV = User(username="developer", kind=UserKind.AGENT)
SCI = User(username="scientist", kind=UserKind.AGENT)
POLL = User(username="pollster", kind=UserKind.AGENT)
RETIRED = User(username="retired", kind=UserKind.AGENT)  # an agent whose role is no longer in the registry
REGISTRY = [DEV, SCI, POLL]
USERS = {user.id: user for user in (ME, ANN, DEV, SCI, POLL, RETIRED)}

NEW_POST = NotificationReason.NEW_POST
THREAD = NotificationReason.THREAD
TOP_LEVEL = NotificationReason.TOP_LEVEL
MENTION = NotificationReason.MENTION


def _post(author: User = ME) -> Message:
    return Message(author_id=author.id, text="Is Rust growing?")


def _reply(parent: Message, author: User) -> Message:
    return Message(post_id=parent.post_id or parent.id, parent_id=parent.id, author_id=author.id, text="text")


def _notified(
    message: Message, earlier: list[Message], post: Message, mentioned: tuple[User, ...] = ()
) -> list[tuple[str, NotificationReason]]:
    """(username, reason) for each notification of `message`, saved after `earlier` on `post`."""
    tree = Tree(post, [*earlier, message]) if message is not post else Tree(post, [])
    author = USERS[message.author_id]
    notifications = notifications_for(
        message, tree, author=author, mentioned=mentioned, agents=REGISTRY, all_agents=[*REGISTRY, RETIRED]
    )
    assert all(n.post_id == post.id and n.message_id == message.id and n.pending for n in notifications)
    return [(USERS[n.user_id].username, n.reason) for n in notifications]


# find_mentions


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("@developer what do you think?", ["developer"]),
        ("What do you think, @Developer?", ["developer"]),
        ("@DEVELOPER and @scientist", ["developer", "scientist"]),
        ("(@developer) [@scientist] \"@pollster\" '@ann'", ["developer", "scientist", "pollster", "ann"]),
        ("@developer's numbers, and @scientist.", ["developer", "scientist"]),
        ("@developer @Developer @developer", ["developer"]),
        ("`@developer` in code", ["developer"]),
        ("@developers", ["developers"]),
        ("@dev_2026", ["dev_2026"]),
        ("write to me@developer.io", []),
        ("https://medium.com/@scientist/post and youtube.com/@pollster", []),
        ("a-@developer, a.@developer, a_@developer, a@@developer, é@developer", []),
        ("@" + "a" * 31, []),
        ("@ alone, and a trailing @", []),
    ],
)
def test_find_mentions(text: str, names: list[str]) -> None:
    assert find_mentions(text) == names


# One rule at a time


def test_a_new_post_notifies_every_agent_in_the_registry() -> None:
    post = _post()

    assert _notified(post, [], post) == [("developer", NEW_POST), ("scientist", NEW_POST), ("pollster", NEW_POST)]


def test_a_reply_notifies_everyone_who_has_commented_in_its_thread() -> None:
    post = _post()
    first = _reply(post, DEV)
    other = _reply(post, SCI)  # another thread
    answer = _reply(first, ME)
    deep = _reply(answer, POLL)
    new = _reply(deep, ANN)

    assert _notified(new, [first, other, answer, deep], post) == [
        ("developer", THREAD),
        ("me", THREAD),
        ("pollster", THREAD),
    ]


def test_a_top_level_comment_notifies_the_post_author_and_the_agents_that_commented_on_the_post() -> None:
    post = _post()
    first = _reply(post, DEV)
    answer = _reply(first, SCI)  # a reply counts as commenting on the post
    mine = _reply(first, ANN)  # a real user who commented is not an agent
    new = _reply(post, ANN)

    assert _notified(new, [first, answer, mine], post) == [
        ("me", TOP_LEVEL),
        ("developer", TOP_LEVEL),
        ("scientist", TOP_LEVEL),
    ]


def test_the_post_author_hears_of_a_reply_only_in_a_thread_it_commented_in() -> None:
    post = _post()
    first = _reply(post, DEV)
    second = _reply(post, SCI)
    mine = _reply(second, ME)

    assert _notified(_reply(first, ANN), [first, second, mine], post) == [("developer", THREAD)]
    assert _notified(_reply(second, ANN), [first, second, mine], post) == [("scientist", THREAD), ("me", THREAD)]


def test_a_mention_by_a_real_user_notifies_the_named_users() -> None:
    post = _post()
    first = _reply(post, DEV)
    new = _reply(first, ME)

    assert _notified(new, [first], post, mentioned=(POLL, ANN)) == [
        ("developer", THREAD),
        ("pollster", MENTION),
        ("ann", MENTION),
    ]


def test_a_mention_by_an_agent_notifies_no_one() -> None:
    post = _post()
    first = _reply(post, DEV)
    new = _reply(first, SCI)

    assert _notified(new, [first], post, mentioned=(POLL, ANN)) == [("developer", THREAD)]


# Several rules, and who is never notified


def test_a_user_gets_one_notification_per_message_and_a_mention_wins() -> None:
    post = _post()
    first = _reply(post, DEV)
    answer = _reply(first, SCI)

    in_thread = _reply(first, ME)
    assert _notified(in_thread, [first, answer], post, mentioned=(DEV,)) == [
        ("developer", MENTION),
        ("scientist", THREAD),
    ]
    top_level = _reply(post, ANN)
    assert _notified(top_level, [first, answer], post, mentioned=(ME, DEV, DEV)) == [
        ("me", MENTION),
        ("developer", MENTION),
        ("scientist", TOP_LEVEL),
    ]
    mentioning_post = _post()
    assert _notified(mentioning_post, [], mentioning_post, mentioned=(SCI,)) == [
        ("developer", NEW_POST),
        ("scientist", MENTION),
        ("pollster", NEW_POST),
    ]


def test_nobody_is_notified_of_their_own_message() -> None:
    post = _post()
    first = _reply(post, DEV)
    mine = _reply(first, ME)

    assert _notified(_reply(mine, DEV), [first, mine], post) == [("me", THREAD)]
    assert _notified(_reply(post, DEV), [first, mine], post) == [("me", TOP_LEVEL)]
    assert _notified(_reply(post, ME), [first, mine], post, mentioned=(ME,)) == [("developer", TOP_LEVEL)]


def test_an_agent_outside_the_registry_is_notified_by_no_rule() -> None:
    post = _post()
    first = _reply(post, RETIRED)
    answer = _reply(first, DEV)

    assert ("retired", NEW_POST) not in _notified(post, [], post)
    assert _notified(_reply(answer, ME), [first, answer], post, mentioned=(RETIRED,)) == [("developer", THREAD)]
    assert _notified(_reply(post, ANN), [first, answer], post) == [("me", TOP_LEVEL), ("developer", TOP_LEVEL)]
