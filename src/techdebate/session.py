"""What a command or the web app works with: the store, the users, the forum, the roles and the tools.

The CLI opens a session per command, and the web app opens one for its whole life, so both see the same users and
write through the same `Forum`."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from techdebate.config import Settings
from techdebate.forum import Forum
from techdebate.models import User
from techdebate.roles import RoleRegistry, default_roles
from techdebate.store import Store, open_store
from techdebate.tools import ToolRegistry, default_tools, make_http_client
from techdebate.users import sync_users


@dataclass(frozen=True)
class Session:
    store: Store
    me: User  # the real user in `username`, who writes from the command line and the web app
    agents: dict[str, User]  # by role name, in registry order
    forum: Forum
    roles: RoleRegistry
    tools: ToolRegistry


@asynccontextmanager
async def open_session(settings: Settings) -> AsyncGenerator[Session]:
    """The store, with the users synced to the role registry, the forum that notifies its agents, and the tools. The
    HTTP client stays open for the whole session, so the agents can use the tools; nothing else makes a request."""
    async with open_store(settings) as store, make_http_client(settings) as client:
        tools = default_tools(settings, client)
        roles = default_roles(tools)
        users = await sync_users(store, roles, settings.username)
        yield Session(
            store=store,
            me=users.me,
            agents=users.agents,
            forum=Forum(store, users.agents.values()),
            roles=roles,
            tools=tools,
        )
