"""Checks a post: every agent looks at it once, in registration order, and may comment on it."""

import logging
from collections.abc import Mapping

from techdebate.analyst import Analyst
from techdebate.config import Settings
from techdebate.llm import LLMClient
from techdebate.models import Message, User
from techdebate.roles import RoleRegistry
from techdebate.tools import ToolRegistry

log = logging.getLogger(__name__)


async def check_post(
    post: Message,
    *,
    agents: Mapping[str, User],
    llm: LLMClient,
    roles: RoleRegistry,
    tools: ToolRegistry,
    settings: Settings,
) -> list[Message]:
    """The comments the agents wrote, in the order they wrote them. `agents` maps each role's name to its user.
    LLMError goes up to the caller."""
    # Built up front, so a bad prompt ID fails before any spending.
    analysts = [(role, Analyst(role, llm, tools, settings)) for role in roles.roles()]
    comments: list[Message] = []

    for role, analyst in analysts:
        decision = await analyst.decide(post.text)
        log.info(
            "agent decided post_id=%s agent=%s speak=%s reason=%s",
            post.id,
            role.name,
            decision.speak,
            decision.reason,
        )
        if decision.speak:
            results = await analyst.gather(decision.queries)
            text = await analyst.write(post.text, results)
            comments.append(Message(post_id=post.id, parent_id=post.id, author_id=agents[role.name].id, text=text))

    return comments
