"""A scripted LLMClient for tests: no network, and every call is recorded."""

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from pydantic import BaseModel, ValidationError

from techdebate.errors import LLMError
from techdebate.llm.base import LLMRequest, LLMResponse, Usage

Reply: TypeAlias = str | BaseModel
Predicate: TypeAlias = Callable[[LLMRequest], bool]


def _last_user_message(request: LLMRequest) -> str:
    return next((m.content for m in reversed(request.messages) if m.role == "user"), "")


class FakeLLM:
    """Replies keyed by the last user message are tried first, then predicate rules in order.

    Like AnthropicLLM, a reply is validated into the request's output_schema (a model reply through its JSON),
    and parsed is set only when the request has a schema.
    """

    def __init__(
        self,
        by_prompt: Mapping[str, Reply] | None = None,
        rules: Sequence[tuple[Predicate, Reply]] = (),
    ) -> None:
        self.by_prompt = dict(by_prompt or {})
        self.rules = list(rules)
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        reply = self._find_reply(request)
        text = reply.model_dump_json() if isinstance(reply, BaseModel) else reply
        parsed: BaseModel | None = None
        schema = request.output_schema
        if schema is not None:
            try:
                parsed = schema.model_validate_json(text)
            except ValidationError as e:
                raise LLMError(f"fake returned output that does not match {schema.__name__}") from e
        return self._response(request, text=text, parsed=parsed)

    def _find_reply(self, request: LLMRequest) -> Reply:
        prompt = _last_user_message(request)
        if prompt in self.by_prompt:
            return self.by_prompt[prompt]
        for matches, reply in self.rules:
            if matches(request):
                return reply
        raise LLMError(f"FakeLLM has no scripted reply for prompt: {prompt!r}")

    @staticmethod
    def _response(request: LLMRequest, text: str, parsed: BaseModel | None = None) -> LLMResponse:
        return LLMResponse(
            text=text,
            parsed=parsed,
            usage=Usage(input_tokens=0, output_tokens=0),
            cost_usd=0.0,
            model="fake",
            prompt_id=request.prompt_id,
        )
