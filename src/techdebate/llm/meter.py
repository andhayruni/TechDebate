"""A client that adds up the cost of every call it passes on, so that new kinds of LLM call are counted without
changing any return type."""

import logging
import time

from techdebate.llm.base import LLMClient, LLMRequest, LLMResponse

log = logging.getLogger(__name__)


class CostMeter:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.total_usd = 0.0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        response = await self._llm.generate(request)
        self.total_usd += response.cost_usd
        log.debug(
            "llm call prompt=%s model=%s input_tokens=%d output_tokens=%d cost=%.6f duration_s=%.1f",
            response.prompt_id,
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.cost_usd,
            time.perf_counter() - start,
        )
        return response
