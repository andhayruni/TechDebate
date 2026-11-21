"""A tool that stores every search it passes on, for one agent's visit to one post, so the trace can show it. It wraps
any tool, as RecordingLLM wraps any LLM client, and returns or raises exactly what the tool does."""

import logging
import time
from typing import Any

from techdebate.errors import NotFoundError, ToolError
from techdebate.models import Evidence, ToolCall, ToolCallOutcome, User
from techdebate.store.base import Store
from techdebate.tools.base import Tool, ToolInput
from techdebate.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class RecordingTool:
    def __init__(self, tool: Tool[Any], store: Store, *, agent: User, post_id: str, visit_id: str) -> None:
        self._tool = tool
        self._store = store
        self._agent = agent
        self._post_id = post_id
        self._visit_id = visit_id

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def input_model(self) -> type[ToolInput]:
        return self._tool.input_model

    async def run(self, input: ToolInput) -> list[Evidence]:
        """Each search is stored once it ends, in a transaction of its own. A cancelled search stores nothing."""
        query = str(getattr(input, "query", ""))
        start = time.perf_counter()
        try:
            evidence = await self._tool.run(input)
        except NotFoundError as e:
            await self._store_call(query, ToolCallOutcome.NO_INFORMATION, start, error=str(e))
            raise
        except ToolError as e:
            await self._store_call(query, ToolCallOutcome.FAILED, start, error=str(e))
            raise
        except Exception as e:  # a bug: stored as a failure, and it goes up as before
            await self._store_call(query, ToolCallOutcome.FAILED, start, error=repr(e))
            raise
        outcome = ToolCallOutcome.EVIDENCE if evidence else ToolCallOutcome.NO_INFORMATION
        await self._store_call(query, outcome, start, evidence=len(evidence))
        return evidence

    async def _store_call(
        self, query: str, outcome: ToolCallOutcome, start: float, *, evidence: int = 0, error: str | None = None
    ) -> None:
        call = ToolCall(
            visit_id=self._visit_id,
            agent_id=self._agent.id,
            post_id=self._post_id,
            tool=self._tool.name,
            query=query,
            outcome=outcome,
            evidence=evidence,
            error=error,
            duration_s=time.perf_counter() - start,
        )
        await self._store.add_tool_call(call)
        log.debug(
            "tool call agent=%s post_id=%s tool=%s query=%r outcome=%s evidence=%d duration_s=%.1f",
            self._agent.username,
            self._post_id,
            call.tool,
            query,
            outcome,
            evidence,
            call.duration_s,
        )


def recording_tools(tools: ToolRegistry, store: Store, *, agent: User, post_id: str, visit_id: str) -> ToolRegistry:
    """The same tools, each storing its searches for the visit."""
    return ToolRegistry(
        RecordingTool(tool, store, agent=agent, post_id=post_id, visit_id=visit_id) for tool in tools.tools()
    )
