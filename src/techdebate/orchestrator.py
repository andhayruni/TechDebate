"""Checks a post: every agent looks at it once, in registration order. An agent that finds the post in its area
researches it, and comments only if its results add something new."""

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
        plan = await analyst.plan(post.text)
        log.info("agent planned post_id=%s agent=%s speak=%s reason=%s", post.id, role.name, plan.speak, plan.reason)
        if not plan.speak:
            continue
        results = await analyst.gather(plan.queries)
        decision = await analyst.decide(post.text, results)
        log.info(
            "agent decided post_id=%s agent=%s reply_to=%s reason=%s",
            post.id,
            role.name,
            decision.reply_to,
            decision.reason,
        )
        for _place in decision.reply_to:  # only the post itself, 0, until 5.4
            text = await analyst.write(post.text, results)
            comments.append(Message(post_id=post.id, parent_id=post.id, author_id=agents[role.name].id, text=text))

    return comments
