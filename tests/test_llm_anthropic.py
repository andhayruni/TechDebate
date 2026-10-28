import json
from collections.abc import Callable
from typing import Any, Literal

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from pydantic import BaseModel, create_model

from techdebate.config import ModelPrice, Settings
from techdebate.errors import LLMError
from techdebate.llm import AnthropicLLM, LLMClient, LLMRequest, Message, ModelTier, Usage
from techdebate.llm.anthropic_client import cost_usd

Handler = Callable[[httpx2.Request], httpx2.Response]


class Answer(BaseModel):
    name: str
    ok: bool
    reason: str


def _settings(**overrides: Any) -> Settings:
    # Ignore a local .env so the tests see the defaults.
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def _llm(handler: Handler, max_retries: int = 0, **overrides: Any) -> AnthropicLLM:
    http_client = DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    client = AsyncAnthropic(api_key="test", max_retries=max_retries, http_client=http_client)
    return AnthropicLLM(_settings(**overrides), client=client)


def _message(text: str, model: str = "claude-haiku-5-5", stop_reason: str = "end_turn", **extra: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1_000, "output_tokens": 200},
        **extra,
    }


def _reply(body: dict[str, Any]) -> Handler:
    return lambda _: httpx2.Response(200, json=body)


def _error(status: int) -> Handler:
    return lambda _: httpx2.Response(status, json={"type": "error", "error": {"type": "error", "message": "boom"}})


def _request(tier: ModelTier = ModelTier.CHEAP, schema: type[BaseModel] | None = None) -> LLMRequest:
    return LLMRequest(
        tier=tier, system="sys", messages=[Message(role="user", content="Rust")], output_schema=schema, max_tokens=500
    )


@pytest.mark.parametrize(
    ("tier", "model", "effort"),
    [(ModelTier.CHEAP, "claude-haiku-5-5", "low"), (ModelTier.STRONG, "claude-sonnet-5-5", "high")],
)
async def test_text_generation_maps_tier_to_model_and_effort(tier: ModelTier, model: str, effort: str) -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=_message("hello", model=model))

    response = await _llm(handler).generate(_request(tier))

    assert sent[0]["model"] == model
    assert sent[0]["output_config"] == {"effort": effort}
    assert sent[0]["system"] == "sys"
    assert sent[0]["messages"] == [{"role": "user", "content": "Rust"}]
    assert response.text == "hello"
    assert response.parsed is None
    assert response.model == model
    assert response.usage == Usage(input_tokens=1_000, output_tokens=200)


async def test_structured_output_is_parsed() -> None:
    answer = Answer(name="builder", ok=True, reason="Widely used")
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=_message(answer.model_dump_json()))

    response = await _llm(handler).generate(_request(schema=Answer))

    output_format = sent[0]["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["required"] == ["name", "ok", "reason"]
    assert response.parsed == answer


async def test_prompt_id_is_copied_but_not_sent() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=_message("hello"))

    request = _request().model_copy(update={"prompt_id": "greet.v1"})
    response = await _llm(handler).generate(request)

    assert response.prompt_id == "greet.v1"
    assert "prompt_id" not in sent[0]


async def test_output_not_matching_schema_raises() -> None:
    llm = _llm(_reply(_message('{"ok": "maybe"}')))
    with pytest.raises(LLMError, match="Answer"):
        await llm.generate(_request(schema=Answer))


def _pick(reply_to: Any) -> type[BaseModel]:
    # A schema built per call, with reply_to limited to the values valid at that moment.
    return create_model("Pick", reply_to=(reply_to, ...), reason=(str, ...))


@pytest.mark.parametrize(
    ("reply_to", "sent_schema", "value"),
    [
        (Literal[1, 2, 3] | None, {"anyOf": [{"type": "integer", "enum": [1, 2, 3]}, {"type": "null"}]}, 2),
        # A one-value Literal stays strict: an enum, not a const moved into the description.
        (Literal[1] | None, {"anyOf": [{"type": "integer", "enum": [1]}, {"type": "null"}]}, 1),
        (None, {"type": "null"}, None),
    ],
)
async def test_per_call_literal_schema(reply_to: Any, sent_schema: dict[str, Any], value: int | None) -> None:
    schema = _pick(reply_to)
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=_message(json.dumps({"reply_to": value, "reason": "r"})))

    response = await _llm(handler).generate(_request(schema=schema))

    sent_reply_to = sent[0]["output_config"]["format"]["schema"]["properties"]["reply_to"]
    assert {k: v for k, v in sent_reply_to.items() if k != "title"} == sent_schema
    assert isinstance(response.parsed, schema)
    assert response.parsed.model_dump() == {"reply_to": value, "reason": "r"}


async def test_per_call_literal_out_of_range_raises() -> None:
    llm = _llm(_reply(_message('{"reply_to": 4, "reason": "r"}')))
    with pytest.raises(LLMError, match="Pick"):
        await llm.generate(_request(schema=_pick(Literal[1] | None)))


async def test_refusal_raises() -> None:
    body = _message("", stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"})
    with pytest.raises(LLMError, match=r"refused.*cyber"):
        await _llm(_reply(body)).generate(_request())


async def test_truncated_structured_output_raises() -> None:
    llm = _llm(_reply(_message('{"name": "buil', stop_reason="max_tokens")))
    with pytest.raises(LLMError, match="max_tokens"):
        await llm.generate(_request(schema=Answer))


async def test_truncated_text_is_returned() -> None:
    response = await _llm(_reply(_message("partial", stop_reason="max_tokens"))).generate(_request())
    assert response.text == "partial"


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "API key"), (404, "unknown model"), (429, "rate limit"), (500, "server error"), (400, "rejected")],
)
async def test_api_errors_become_llm_error(status: int, message: str) -> None:
    with pytest.raises(LLMError, match=message):
        await _llm(_error(status)).generate(_request())


async def test_server_errors_are_retried() -> None:
    responses = iter([_error(500), _reply(_message("ok"))])
    response = await _llm(lambda r: next(responses)(r), max_retries=1).generate(_request())
    assert response.text == "ok"


async def test_cost_uses_price_table() -> None:
    response = await _llm(_reply(_message("hi", model="claude-sonnet-5-5"))).generate(_request(ModelTier.STRONG))
    # 1,000 input at $2/MTok + 200 output at $10/MTok
    assert response.cost_usd == pytest.approx(0.004)


def test_cost_usd() -> None:
    price = ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50)
    assert cost_usd(Usage(input_tokens=2_000_000, output_tokens=1_000_000), price) == pytest.approx(0.70)


def test_model_without_price_fails_at_construction() -> None:
    with pytest.raises(LLMError, match="claude-unknown"):
        _llm(_reply(_message("x")), strong_model="claude-unknown")


def test_satisfies_protocol() -> None:
    client: LLMClient = _llm(_reply(_message("x")))
    assert client is not None
