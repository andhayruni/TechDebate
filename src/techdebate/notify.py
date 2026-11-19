"""Who is notified of a new message. These are all the rules:

- A new post notifies every agent.
- A new comment notifies everyone who has commented in its thread.
- A new top-level comment also notifies the post's author, and every agent that has commented on the post.
- A message written by a real user notifies every user it names with @username. An agent's @name is plain text.

Nobody is notified of their own message, and a user gets one notification per message, whose reason is `mention` when
several rules apply. Only the agents in the role registry are notified.
"""

import re
from collections.abc import Iterable

from techdebate.models import USERNAME_PATTERN, Message, Notification, NotificationReason, User, UserKind
from techdebate.tree import Tree

# An @ counts only where it cannot be part of a word, a URL path or an email address. The name runs as far as the
# username characters go, in either case, so "@developers" is the name "developers".
_MENTION = re.compile(r"(?<![\w/.\-@])@([A-Za-z0-9_]+)")


def find_mentions(text: str) -> list[str]:
    """The usernames a text names with @, lowercased, each once, in the order they first appear. Whether they are
    users is up to the caller."""
    names = (match.group(1).lower() for match in _MENTION.finditer(text))
    return list(dict.fromkeys(name for name in names if re.fullmatch(USERNAME_PATTERN, name)))


def notifications_for(
    message: Message,
    tree: Tree,
    *,
    author: User,
    mentioned: Iterable[User],
    agents: Iterable[User],
    all_agents: Iterable[User],
) -> list[Notification]:
    """The notifications of a new message. `tree` is its post with the message in it (a post: no comments), `mentioned`
    the users found for its @names, `agents` the agents in the registry, and `all_agents` every agent user, so that an
    agent outside the registry is told apart from a real user."""
    registry = [agent.id for agent in agents]
    left_out = {agent.id for agent in all_agents} - set(registry)
    reasons: dict[str, NotificationReason] = {}  # by user ID, in the order they were found

    def notify(user_id: str, reason: NotificationReason) -> None:
        if user_id != author.id and user_id not in left_out:
            reasons.setdefault(user_id, reason)

    if message.parent_id is None:
        for agent_id in registry:
            notify(agent_id, NotificationReason.NEW_POST)
    else:
        for comment in tree.thread(message):
            notify(comment.author_id, NotificationReason.THREAD)
        if message.parent_id == tree.post.id:  # a top-level comment, whose thread is only itself
            notify(tree.post.author_id, NotificationReason.TOP_LEVEL)
            commenters = {comment.author_id for comment in tree.comments}
            for agent_id in registry:
                if agent_id in commenters:
                    notify(agent_id, NotificationReason.TOP_LEVEL)

    if author.kind is UserKind.REAL:
        for user in mentioned:
            notify(user.id, NotificationReason.MENTION)
            if user.id in reasons:  # a mention wins over any other rule
                reasons[user.id] = NotificationReason.MENTION

    return [
        Notification(user_id=user_id, post_id=tree.post.id, message_id=message.id, reason=reason)
        for user_id, reason in reasons.items()
    ]
