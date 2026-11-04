"""Runs a debate. For now that is at most one turn: roles act in registration order until one speaks."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from techdebate.analyst import Analyst
from techdebate.config import Settings
from techdebate.llm import LLMClient, LLMRequest, LLMResponse
from techdebate.models import Debate, DebateStatus
from techdebate.roles import RoleRegistry
from techdebate.tools import ToolRegistry

log = logging.getLogger(__name__)


class _CostMeter:
    """Adds up the cost of every call, so that new kinds of LLM call are counted without changing any return type."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.total_usd = 0.0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await self._llm.generate(request)
        self.total_usd += response.cost_usd
        return response


async def run_debate(
    topic: str,
    *,
    llm: LLMClient,
    roles: RoleRegistry,
    tools: ToolRegistry,
    settings: Settings,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Debate:
    """A finished debate with no turns means every role passed. ToolError and LLMError go up to the caller."""
    rng = rng if rng is not None else random.Random()
    meter = _CostMeter(llm)
    # Built up front, so a bad prompt ID fails before any waiting or spending.
    analysts = [(role, Analyst(role, meter, tools, settings)) for role in roles.roles()]
    debate = Debate(topic=topic)

    for role, analyst in analysts:
        if settings.read_delays:
            await sleep(rng.uniform(settings.read_delay_min_s, settings.read_delay_max_s))
        decision = await analyst.decide(topic)
        log.info("role decided role=%s speak=%s reason=%s", role.name, decision.speak, decision.reason)
        if decision.speak:
            evidence = await analyst.gather(topic)
            debate.turns.append(await analyst.write(topic, evidence))
            break

    debate.status = DebateStatus.FINISHED
    debate.total_cost_usd = meter.total_usd
    return debate
