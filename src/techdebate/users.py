"""Users: the username rule, the sync that makes the users table match the role registry and the configured username,
and the name that is shown for a user."""

import re
from dataclasses import dataclass

from techdebate.errors import UserError
from techdebate.models import USERNAME_PATTERN, User, UserKind
from techdebate.roles import RoleRegistry, display_name
from techdebate.store import Store, StoreOps


@dataclass(frozen=True)
class Users:
    """The real user who writes from the command line, and the agents that take part."""

    me: User
    agents: dict[str, User]  # by role name, in registry order


def check_username(username: str) -> str:
    """The username, when it follows the rule. Raises UserError otherwise, rather than guessing a fix."""
    if re.fullmatch(USERNAME_PATTERN, username) is None:
        raise UserError(f"username {username!r} is invalid: use 1 to 30 lowercase letters, digits or underscores")
    return username


async def sync_users(store: Store, roles: RoleRegistry, username: str) -> Users:
    """Adds whoever is missing and reuses whoever is already there: an agent for every role in the registry, and the
    real user with this username. It never removes a user, so an agent whose role is left out keeps its comments, but
    it is not returned. Raises UserError on a bad username or a clash between the two kinds, and then saves nothing."""
    check_username(username)
    # One transaction, so two processes starting at the same time create each user once.
    async with store.transaction() as tx:
        agents = {role.name: await _ensure(tx, role.name, UserKind.AGENT) for role in roles.roles()}
        me = await _ensure(tx, username, UserKind.REAL)
    return Users(me=me, agents=agents)


async def _ensure(tx: StoreOps, username: str, kind: UserKind) -> User:
    user = await tx.find_user(username)
    if user is None:
        user = User(username=username, kind=kind)
        await tx.add_user(user)
    elif user.kind is not kind:
        if kind is UserKind.REAL:
            raise UserError(f"username {username!r} belongs to an agent, so a real user cannot have it")
        raise UserError(f"the {username} agent cannot be added: a real user already has the username {username!r}")
    return user


def shown_name(user: User) -> str:
    """The name printed for a user. An agent's name always carries the AI label."""
    if user.kind is UserKind.AGENT:
        return f"{display_name(user.username)} [AI]"
    return user.username
