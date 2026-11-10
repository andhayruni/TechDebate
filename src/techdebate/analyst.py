"""An analyst does an agent's work: it decides whether to comment and what to search for, gathers evidence and writes
the comment."""

import asyncio
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, create_model

from techdebate.config import Settings
from techdebate.errors import LLMError, NotFoundError, ToolError
from techdebate.llm import LLMClient, LLMRequest, Message, ModelTier
from techdebate.models import Evidence, SpeakDecision
from techdebate.prompts import load_prompt_by_id
from techdebate.roles import RoleSpec
from techdebate.tools import ToolRegistry, format_evidence

log = logging.getLogger(__name__)

_DECIDE_PROMPT_ID = "decide.v1"  # shared by every role


@dataclass(frozen=True)
class ToolResult:
    """What one query found. With no evidence, the query found no information, or the tool failed."""

    tool: str
    query: str | None  # None: the tool was not used
    evidence: list[Evidence] = field(default_factory=list[Evidence])
    failed: bool = False


def _decision_schema(tool_names: Iterable[str]) -> type[SpeakDecision]:
    """The schema the model fills in: every one of the role's tools gets a list of queries, and no other tool does."""
    tool_fields: dict[str, Any] = {name: (list[str], ...) for name in tool_names}
    queries = create_model("ToolQueries", __config__=ConfigDict(extra="forbid"), **tool_fields)
    return create_model("SpeakDecision", __base__=SpeakDecision, queries=(queries, ...))


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
        self._decide_prompt = load_prompt_by_id(_DECIDE_PROMPT_ID)
        self._decision_schema = _decision_schema(role.tool_names)
        self._tool_descriptions = [self._describe_tool(name) for name in role.tool_names]

    async def decide(self, topic: str) -> SpeakDecision:
        request = LLMRequest(
            tier=ModelTier.CHEAP,
            system=self._decide_prompt.render(
                display_name=self._role.display_name,
                question=self._role.question,
                tools=self._tool_descriptions,
                max_queries_per_tool=self._max_queries_per_tool,
            ),
            messages=[Message(role="user", content=f"Topic: {topic}")],
            output_schema=self._decision_schema,
            max_tokens=self._decide_max_tokens,
            prompt_id=self._decide_prompt.id,
        )
        response = await self._llm.generate(request)
        # A client returns output_schema's type, or raises. Its nested queries become the plain mapping.
        return SpeakDecision.model_validate(cast("BaseModel", response.parsed).model_dump())

    async def gather(self, queries: Mapping[str, list[str]]) -> list[ToolResult]:
        """One result per query, in the role's tool order, and one for each tool with no queries. It never raises for
        a failed tool: the write prompt is told what happened."""
        per_tool = await asyncio.gather(
            *(self._run_tool(name, self._clean(queries.get(name, []))) for name in self._role.tool_names)
        )
        return [result for results in per_tool for result in results]

    async def write(self, topic: str, results: list[ToolResult]) -> str:
        """The text of the comment."""
        # The instructions are the system prompt. Tool results hold text written by strangers, so they stay out of it.
        request = LLMRequest(
            tier=ModelTier.STRONG,
            system=self._write_prompt.render(display_name=self._role.display_name, question=self._role.question),
            messages=[Message(role="user", content=_user_message(topic, results))],
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


def _user_message(topic: str, results: list[ToolResult]) -> str:
    return f"Topic: {topic}\n\nTool results:\n\n" + "\n\n".join(_report(result) for result in results)
