"""LLMClient backed by the Anthropic Messages API."""

import logging
from typing import Any, cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageParam, OutputConfigParam
from pydantic import BaseModel, ValidationError

from techdebate.config import Effort, ModelPrice, Settings
from techdebate.errors import LLMError
from techdebate.llm.base import LLMRequest, LLMResponse, ModelTier, Usage

log = logging.getLogger(__name__)


def cost_usd(usage: Usage, price: ModelPrice) -> float:
    return (usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok) / 1_000_000


class AnthropicLLM:
    def __init__(self, settings: Settings, client: AsyncAnthropic | None = None) -> None:
        self._models: dict[ModelTier, tuple[str, Effort]] = {
            ModelTier.CHEAP: (settings.cheap_model, settings.cheap_effort),
            ModelTier.STRONG: (settings.strong_model, settings.strong_effort),
        }
        missing = sorted({m for m, _ in self._models.values() if m not in settings.model_prices})
        if missing:
            raise LLMError(f"no price configured for model(s): {', '.join(missing)}")
        self._prices = settings.model_prices

        if client is None:
            key = settings.anthropic_api_key
            # With no key, the SDK resolves credentials itself (ANTHROPIC_AUTH_TOKEN, an `ant` profile, ...).
            client = AsyncAnthropic(
                api_key=key.get_secret_value() if key else None,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
        self._client = client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model, effort = self._models[request.tier]
        output_config: OutputConfigParam = {"effort": effort}
        if request.output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic.transform_schema(_const_to_enum(request.output_schema.model_json_schema())),
            }
        messages: list[MessageParam] = [{"role": m.role, "content": m.content} for m in request.messages]

        try:
            message = await self._client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=messages,
                output_config=output_config,
            )
        except anthropic.APIError as e:
            raise _to_llm_error(e, model) from e

        return self._response(message, model, request)

    def _response(self, message: AnthropicMessage, requested_model: str, request: LLMRequest) -> LLMResponse:
        model = message.model  # the model that answered
        text = "".join(block.text for block in message.content if block.type == "text")
        # Worked out first: a call the provider answered is paid for, even when it fails below.
        usage = Usage(input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens)
        cost = cost_usd(usage, self._price(requested_model, model))

        def failed(reason: str) -> LLMError:
            return LLMError(reason, model=model, usage=usage, cost_usd=cost, text=text)

        if message.stop_reason == "refusal":
            category = message.stop_details.category if message.stop_details else None
            raise failed(f"{model} refused the request (category: {category or 'unspecified'})")

        parsed: BaseModel | None = None
        schema = request.output_schema
        if schema is not None:
            if message.stop_reason == "max_tokens":
                raise failed(f"{model} hit max_tokens before finishing the {schema.__name__} output")
            try:
                parsed = schema.model_validate_json(text)
            except ValidationError as e:
                raise failed(f"{model} returned output that does not match {schema.__name__}") from e
        elif message.stop_reason == "max_tokens":
            log.warning("llm output truncated model=%s max_tokens reached", model)

        return LLMResponse(
            text=text, parsed=parsed, usage=usage, cost_usd=cost, model=model, prompt_id=request.prompt_id
        )

    def _price(self, requested_model: str, answered_model: str) -> ModelPrice:
        """The answering model's price. When it has none, the requested model's, which always has one: the call is
        already paid for, so it is not failed."""
        if answered_model in self._prices:
            return self._prices[answered_model]
        log.warning("llm priced by the requested model requested=%s answered=%s", requested_model, answered_model)
        return self._prices[requested_model]


def _const_to_enum(node: Any) -> Any:
    # Pydantic writes a one-value Literal as "const", which transform_schema moves into the description,
    # leaving the field unconstrained. A one-value "enum" is kept.
    if isinstance(node, dict):
        node = {key: _const_to_enum(value) for key, value in cast("dict[str, Any]", node).items()}
        if "const" in node and not isinstance(node["const"], dict):
            node["enum"] = [node.pop("const")]
        return node
    if isinstance(node, list):
        return [_const_to_enum(item) for item in cast("list[Any]", node)]
    return node


def _to_llm_error(e: anthropic.APIError, model: str) -> LLMError:
    """The provider gave no answer, so the error carries only the model that was requested."""
    return LLMError(_api_error_message(e, model), model=model)


def _api_error_message(e: anthropic.APIError, model: str) -> str:
    # Most specific first: retries have already run for rate limits, 5xx and connection errors.
    if isinstance(e, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return "Anthropic API key is missing or invalid (set ANTHROPIC_API_KEY)"
    if isinstance(e, anthropic.NotFoundError):
        return f"unknown model: {model}"
    if isinstance(e, anthropic.RateLimitError):
        return "Anthropic rate limit reached, still failing after retries"
    if isinstance(e, anthropic.APIStatusError):
        if e.status_code >= 500:
            return f"Anthropic server error ({e.status_code}), still failing after retries"
        return f"Anthropic rejected the request ({e.status_code}): {e.message}"
    if isinstance(e, anthropic.APITimeoutError):
        return "Anthropic request timed out"
    if isinstance(e, anthropic.APIConnectionError):
        return "could not reach the Anthropic API"
    return f"Anthropic API error: {e}"
