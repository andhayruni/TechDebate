"""Checks a post: every agent looks at it once, in registration order. An agent that finds the post in its area
researches it, and comments only if its results add something new."""

import logging
from collections.abc import Mapping

from techdebate.analyst import Analyst, News
from techdebate.config import Settings
from techdebate.llm import LLMClient
from techdebate.models import Message, User
from techdebate.places import POST, one_per_thread, place_message, places_for
from techdebate.roles import RoleRegistry
from techdebate.tools import ToolRegistry
from techdebate.tree import Tree
from techdebate.views import overview_for

log = logging.getLogger(__name__)


async def check_post(
    post: Message,
    *,
    author: User,
    agents: Mapping[str, User],
    llm: LLMClient,
    roles: RoleRegistry,
    tools: ToolRegistry,
    settings: Settings,
) -> list[Message]:
    """The comments the agents wrote, in the order they wrote them. `author` wrote the post, and `agents` maps each
    role's name to its user. LLMError goes up to the caller."""
    # Built up front, so a bad prompt ID fails before any spending.
    analysts = [(role, Analyst(role, llm, tools, settings)) for role in roles.roles()]
    # Every agent sees the post alone, new to it: the comments written here are not shown to the agents after, since
    # an agent can't write a reply yet, only a top-level comment.
    tree = Tree(post, [])
    users = {user.id: user for user in (author, *agents.values())}
    overview = overview_for(tree, users, reply_chars=settings.overview_reply_chars)
    news = News(new=(POST,))
    comments: list[Message] = []

    for role, analyst in analysts:
        agent = agents[role.name]
        plan = await analyst.plan(overview, news)
        log.info("agent planned post_id=%s agent=%s speak=%s reason=%s", post.id, role.name, plan.speak, plan.reason)
        if not plan.speak:
            continue
        results = await analyst.gather(plan.queries)
        places = places_for(tree, agent.id, max_per_thread=settings.max_comments_per_thread)
        decision = await analyst.decide(overview, results, places, news)
        reply_to = one_per_thread(tree, decision.reply_to)
        log.info(
            "agent decided post_id=%s agent=%s reply_to=%s reason=%s",
            post.id,
            role.name,
            [item.place for item in reply_to],
            decision.reason,
        )
        for item in reply_to:  # only the post itself, 0, until writing a reply comes
            parent = place_message(tree, item.place)
            text = await analyst.write(post.text, results, item.point)
            comments.append(Message(post_id=post.id, parent_id=parent.id, author_id=agent.id, text=text))

    return comments
