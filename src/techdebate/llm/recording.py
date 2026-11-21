"""A client that stores every call it passes on, for one agent's visit to one post, so the budgets and the trace can
read them."""

import logging
import time

from techdebate.errors import LLMError
from techdebate.llm.base import LLMClient, LLMRequest, LLMResponse
from techdebate.models import LLMCall, User
from techdebate.store.base import Store

log = logging.getLogger(__name__)

# The model of a failed call whose client gave none.
_UNKNOWN_MODEL = "unknown"


class RecordingLLM:
    def __init__(self, llm: LLMClient, store: Store, *, agent: User, post_id: str, visit_id: str) -> None:
        self._llm = llm
        self._store = store
        self._agent = agent
        self._post_id = post_id
        self._visit_id = visit_id
        self.total_usd = 0.0  # the cost of the calls it stored, failed ones included

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Every call is stored with its answer: one that returns, and one that fails with LLMError, with its error,
        and with its tokens, cost and text when the provider answered. The error then goes up unchanged."""
        if request.prompt_id is None:
            # Every agent call loads its prompt from a file, which gives it an ID.
            raise ValueError("an agent's AI call needs a prompt ID, to be stored")
        start = time.perf_counter()
        try:
            response = await self._llm.generate(request)
        except LLMError as e:
            usage = e.usage
            await self._store_call(
                LLMCall(
                    agent_id=self._agent.id,
                    post_id=self._post_id,
                    visit_id=self._visit_id,
                    prompt_id=request.prompt_id,
                    model=e.model or _UNKNOWN_MODEL,
                    input_tokens=usage.input_tokens if usage else 0,
                    output_tokens=usage.output_tokens if usage else 0,
                    cost_usd=e.cost_usd,
                    duration_s=time.perf_counter() - start,
                    output=e.text,
                    error=str(e),
                )
            )
            raise
        await self._store_call(
            LLMCall(
                agent_id=self._agent.id,
                post_id=self._post_id,
                visit_id=self._visit_id,
                prompt_id=request.prompt_id,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=response.cost_usd,
                duration_s=time.perf_counter() - start,
                output=response.text,
            )
        )
        return response

    async def _store_call(self, call: LLMCall) -> None:
        # Saved after the call, in a transaction of its own: no AI call is made inside a transaction.
        await self._store.add_llm_call(call)
        self.total_usd += call.cost_usd
        log.debug(
            "llm call agent=%s post_id=%s prompt=%s model=%s input_tokens=%d output_tokens=%d cost=%.6f "
            "duration_s=%.1f%s",
            self._agent.username,
            self._post_id,
            call.prompt_id,
            call.model,
            call.input_tokens,
            call.output_tokens,
            call.cost_usd,
            call.duration_s,
            "" if call.error is None else f" error={call.error}",
        )
