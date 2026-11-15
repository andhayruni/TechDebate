"""An analyst does an agent's work on a post: it plans its research, gathers evidence, decides whether to comment, and
writes the comment."""

import asyncio
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import GenericAlias
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, WithJsonSchema, create_model

from techdebate.config import Settings
from techdebate.errors import LLMError, NotFoundError, ToolError
from techdebate.llm import LLMClient, LLMRequest, Message, ModelTier
from techdebate.models import ReplyTo, ResearchPlan, SpeakDecision, ToolResult
from techdebate.places import POST, Places
from techdebate.prompts import load_prompt_by_id
from techdebate.roles import RoleSpec
from techdebate.tools import ToolRegistry, format_evidence

log = logging.getLogger(__name__)

# Shared by every role.
_RESEARCH_PROMPT_ID = "research.v1"
_DECIDE_PROMPT_ID = "decide.v1"


@dataclass(frozen=True)
class News:
    """What is new to an agent on a post, by place (0 for the post, n for comment #n): the messages its notifications
    are about, and the ones in which a user asked it something directly, by mentioning it."""

    new: tuple[int, ...]
    asked_in: tuple[int, ...] = ()


def _plan_schema(tool_names: Iterable[str]) -> type[ResearchPlan]:
    """The schema the model fills in: every one of the role's tools gets a list of queries, and no other tool does."""
    tool_fields: dict[str, Any] = {name: (list[str], ...) for name in tool_names}
    queries = create_model("ToolQueries", __config__=ConfigDict(extra="forbid"), **tool_fields)
    return create_model("ResearchPlan", __base__=ResearchPlan, queries=(queries, ...))


def _decision_schema(places: Sequence[int]) -> type[SpeakDecision]:
    """The schema the model fills in: reply_to can name only the places given, so a closed place can't be picked."""
    # An enum, not the const pydantic gives a one-value Literal: the provider's transform moves a const into the
    # description, and the model would no longer be held to it.
    place = Annotated[cast("Any", Literal)[tuple(places)], WithJsonSchema({"type": "integer", "enum": list(places)})]
    place_field: dict[str, Any] = {"place": (place, ...)}
    reply_to = create_model("ReplyTo", __base__=ReplyTo, **place_field)
    fields: dict[str, Any] = {"reply_to": (GenericAlias(list, reply_to), ...)}  # list[ReplyTo], place in places
    return create_model("SpeakDecision", __base__=SpeakDecision, **fields)


class Analyst:
    def __init__(self, role: RoleSpec, llm: LLMClient, tools: ToolRegistry, settings: Settings) -> None:
        self._role = role
        self._llm = llm
        self._tools = tools
        self._write_max_tokens = settings.write_max_tokens
        self._decide_max_tokens = settings.decide_max_tokens
        self._max_queries_per_tool = settings.max_queries_per_tool
        # Loaded here, so a bad ID fails before any tool or LLM call.
        self._write_prompt = load_prompt_by_id(role.prompt_id)
        self._research_prompt = load_prompt_by_id(_RESEARCH_PROMPT_ID)
        self._decide_prompt = load_prompt_by_id(_DECIDE_PROMPT_ID)
        self._plan_schema = _plan_schema(role.tool_names)
        self._tool_descriptions = [self._describe_tool(name) for name in role.tool_names]

    async def plan(self, overview: str, news: News) -> ResearchPlan:
        """The first look at a post, from its overview: whether to research it, and the queries for each tool."""
        request = LLMRequest(
            tier=ModelTier.CHEAP,
            system=self._research_prompt.render(
                display_name=self._role.display_name,
                question=self._role.question,
                tools=self._tool_descriptions,
                max_queries_per_tool=self._max_queries_per_tool,
            ),
            messages=[Message(role="user", content=f"{overview}\n\n{_news(news)}")],
            output_schema=self._plan_schema,
            max_tokens=self._decide_max_tokens,
            prompt_id=self._research_prompt.id,
        )
        response = await self._llm.generate(request)
        # A client returns output_schema's type, or raises. Its nested queries become the plain mapping.
        return ResearchPlan.model_validate(cast("BaseModel", response.parsed).model_dump())

    async def decide(self, overview: str, results: list[ToolResult], places: Places, news: News) -> SpeakDecision:
        """Where to comment, after the research: only places open to the agent, each with its point. An empty reply_to
        is a pass. Raises ValueError when no place is open, since the call would have nothing to choose from."""
        if not places.open:
            raise ValueError(f"{self._role.name}: no place is open, so there is nothing to decide")
        content = "\n\n".join([overview, f"{_news(news)}\n{_places(places)}", _tool_report(results)])
        request = LLMRequest(
            tier=ModelTier.CHEAP,
            system=self._decide_prompt.render(display_name=self._role.display_name, question=self._role.question),
            messages=[Message(role="user", content=content)],
            output_schema=_decision_schema(places.open),
            max_tokens=self._decide_max_tokens,
            prompt_id=self._decide_prompt.id,
        )
        response = await self._llm.generate(request)
        decision = SpeakDecision.model_validate(cast("BaseModel", response.parsed).model_dump())
        # A place named twice is kept once, with its first point: the API doesn't enforce unique items.
        kept: dict[int, ReplyTo] = {}
        for item in decision.reply_to:
            kept.setdefault(item.place, item)
        return decision.model_copy(update={"reply_to": list(kept.values())})

    async def gather(self, queries: Mapping[str, list[str]]) -> list[ToolResult]:
        """One result per query, in the role's tool order, and one for each tool with no queries. It never raises for
        a failed tool: the write prompt is told what happened."""
        per_tool = await asyncio.gather(
            *(self._run_tool(name, self._clean(queries.get(name, []))) for name in self._role.tool_names)
        )
        return [result for results in per_tool for result in results]

    async def write(self, post: str, results: list[ToolResult], point: str) -> str:
        """The text of the comment, built around `point`: the reason the decide call gave for commenting."""
        # The instructions are the system prompt. The post, the tool results and the point (model output shaped by
        # both) are content, so they stay out of it.
        request = LLMRequest(
            tier=ModelTier.STRONG,
            system=self._write_prompt.render(display_name=self._role.display_name, question=self._role.question),
            messages=[Message(role="user", content=f"Post: {post}\n\n{_tool_report(results)}\n\nYour point: {point}")],
            max_tokens=self._write_max_tokens,
            prompt_id=self._write_prompt.id,
        )
        response = await self._llm.generate(request)
        if not response.text.strip():
            # Thinking counts against max_tokens, so a call can use it all and return no text.
            raise LLMError(
                f"{self._role.name}: {response.model} returned a blank comment (write_max_tokens may be too low)"
            )
        return response.text

    def _describe_tool(self, name: str) -> dict[str, str]:
        tool = self._tools.get(name)
        query = tool.input_model.model_fields["query"].description or "A search query"
        return {"name": name, "description": tool.description, "query": query}

    def _clean(self, queries: list[str]) -> list[str]:
        """Without blanks and duplicates, and within the cap: the API doesn't enforce a maximum list length."""
        stripped = (query.strip() for query in queries)
        return list(dict.fromkeys(query for query in stripped if query))[: self._max_queries_per_tool]

    async def _run_tool(self, name: str, queries: list[str]) -> list[ToolResult]:
        if not queries:
            return [ToolResult(tool=name, query=None)]
        return list(await asyncio.gather(*(self._call_tool(name, query) for query in queries)))

    async def _call_tool(self, name: str, query: str) -> ToolResult:
        try:
            tool = self._tools.get(name)
            return ToolResult(tool=name, query=query, evidence=await tool.run(tool.input_model(query=query)))
        except NotFoundError as e:
            log.info("tool found nothing role=%s tool=%s query=%r error=%s", self._role.name, name, query, e)
            return ToolResult(tool=name, query=query)
        except ToolError as e:
            log.warning("tool failed role=%s tool=%s query=%r error=%s", self._role.name, name, query, e)
            return ToolResult(tool=name, query=query, failed=True)


def _report(result: ToolResult) -> str:
    if result.query is None:
        return f"{result.tool}: not used"
    heading = f'{result.tool}, query "{result.query}"'
    if result.failed:
        return f"{heading}: the tool failed"
    if not result.evidence:
        return f"{heading}: no information found"
    return f"{heading}:\n{format_evidence(result.evidence)}"


def _tool_report(results: list[ToolResult]) -> str:
    """One entry per query: what the decide and the write calls both see."""
    return "Tool results:\n\n" + "\n\n".join(_report(result) for result in results)


def _news(news: News) -> str:
    """What is new to the agent, by number, as the overview shows it."""
    lines = [f"New to you: {', '.join(_place_name(place) for place in news.new)}."] if news.new else []
    if news.asked_in:
        lines.append(f"A user asked you directly in {_and(_place_name(place) for place in news.asked_in)}.")
    return "\n".join(lines)


def _places(places: Places) -> str:
    """Where the agent can comment, and why the places it can't pick are closed."""
    names = [f"{POST} (a top-level comment on the post)" if p == POST else f"#{p}" for p in places.open]
    lines = [f"You can comment on: {', '.join(names)}."]
    if places.has_top_level:
        lines.append("You already have a top-level comment on this post, so you can only reply.")
    if places.full_threads:
        threads = "thread" if len(places.full_threads) == 1 else "threads"
        numbers = _and(f"#{n}" for n in places.full_threads)
        lines.append(f"You have reached your limit of {places.limit} comments in {threads} {numbers}.")
    return "\n".join(lines)


def _place_name(place: int) -> str:
    return "the post" if place == POST else f"#{place}"


def _and(names: Iterable[str]) -> str:
    """`a`, `a and b`, or `a, b and c`."""
    *rest, last = names
    return f"{', '.join(rest)} and {last}" if rest else last
