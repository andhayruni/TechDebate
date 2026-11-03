"""An analyst does a role's work: it gathers evidence with the role's tools and writes turns from it."""

import asyncio
import logging

from techdebate.config import Settings
from techdebate.errors import ToolError
from techdebate.llm import LLMClient, LLMRequest, Message, ModelTier
from techdebate.models import Evidence, Turn
from techdebate.prompts import load_prompt_by_id
from techdebate.roles import RoleSpec
from techdebate.tools import ToolRegistry, format_evidence

log = logging.getLogger(__name__)


class Analyst:
    def __init__(self, role: RoleSpec, llm: LLMClient, tools: ToolRegistry, settings: Settings) -> None:
        self._role = role
        self._llm = llm
        self._tools = tools
        self._write_max_tokens = settings.write_max_tokens
        self._prompt = load_prompt_by_id(role.prompt_id)  # here, so a bad ID fails before any tool call

    async def gather(self, topic: str) -> list[Evidence]:
        """Evidence from every tool that worked, in the role's tool order. Raises ToolError if all of them failed."""
        results = await asyncio.gather(*(self._call_tool(name, topic) for name in self._role.tool_names))
        evidence: list[Evidence] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, ToolError):
                errors.append(str(result))
            else:
                evidence.extend(result)
        if errors and len(errors) == len(results):
            raise ToolError(f"{self._role.name}: every tool failed: {'; '.join(errors)}")
        return evidence

    async def write(self, topic: str, evidence: list[Evidence]) -> Turn:
        # The instructions are the system prompt. Evidence holds text written by strangers, so it stays out of it.
        request = LLMRequest(
            tier=ModelTier.STRONG,
            system=self._prompt.render(display_name=self._role.display_name, question=self._role.question),
            messages=[Message(role="user", content=_user_message(topic, evidence))],
            max_tokens=self._write_max_tokens,
            prompt_id=self._prompt.id,
        )
        response = await self._llm.generate(request)
        return Turn(analyst=self._role.name, text=response.text, cost_usd=response.cost_usd)

    async def _call_tool(self, name: str, topic: str) -> list[Evidence] | ToolError:
        try:
            tool = self._tools.get(name)
            return await tool.run(tool.input_model(query=topic))
        except ToolError as e:
            log.warning("tool failed role=%s tool=%s error=%s", self._role.name, name, e)
            return e


def _user_message(topic: str, evidence: list[Evidence]) -> str:
    return f"Topic: {topic}\n\nEvidence:\n\n{format_evidence(evidence) or 'None found.'}"
