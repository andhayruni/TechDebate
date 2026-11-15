"""A client that stores every call it passes on, for one agent and one post, so the budgets and the trace can read
them."""

import logging
import time

from techdebate.llm.base import LLMClient, LLMRequest, LLMResponse
from techdebate.models import LLMCall, User
from techdebate.store.base import Store

log = logging.getLogger(__name__)


class RecordingLLM:
    def __init__(self, llm: LLMClient, store: Store, *, agent: User, post_id: str) -> None:
        self._llm = llm
        self._store = store
        self._agent = agent
        self._post_id = post_id

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Only a call that returns is stored: a failed one raises before there is anything to store."""
        if request.prompt_id is None:
            # Every agent call loads its prompt from a file, which gives it an ID.
            raise ValueError("an agent's AI call needs a prompt ID, to be stored")
        start = time.perf_counter()
        response = await self._llm.generate(request)
        duration_s = time.perf_counter() - start
        # Saved after the call, in a transaction of its own: no AI call is made inside a transaction.
        await self._store.add_llm_call(
            LLMCall(
                agent_id=self._agent.id,
                post_id=self._post_id,
                prompt_id=request.prompt_id,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=response.cost_usd,
                duration_s=duration_s,
            )
        )
        log.debug(
            "llm call agent=%s post_id=%s prompt=%s model=%s input_tokens=%d output_tokens=%d cost=%.6f "
            "duration_s=%.1f",
            self._agent.username,
            self._post_id,
            request.prompt_id,
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.cost_usd,
            duration_s,
        )
        return response
