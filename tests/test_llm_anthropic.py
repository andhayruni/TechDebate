import json
import logging
from collections.abc import Callable
from typing import Any, Literal

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from pydantic import BaseModel, create_model

from techdebate.analyst import _decision_schema  # pyright: ignore[reportPrivateUsage]
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


@pytest.mark.parametrize("places", [[0], [0, 3, 5]])
async def test_per_call_list_of_places_stays_strict(places: list[int]) -> None:
    # A plain Literal[0] would send {"const": 0}, which the transform moves into the description, so the decision
    # schema gives each item's place an enum.
    schema = _decision_schema(places)
    reply = {"reason": "r", "reply_to": [{"place": place, "point": f"p{place}"} for place in places]}
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=_message(json.dumps(reply)))

    response = await _llm(handler).generate(_request(schema=schema))

    sent_schema = sent[0]["output_config"]["format"]["schema"]
    sent_reply_to = sent_schema["properties"]["reply_to"]
    assert sent_reply_to["type"] == "array"
    assert sent_reply_to["items"] == {"$ref": "#/$defs/ReplyTo"}
    place = sent_schema["$defs"]["ReplyTo"]["properties"]["place"]
    assert {k: v for k, v in place.items() if k != "title"} == {"type": "integer", "enum": places}
    assert isinstance(response.parsed, schema)
    assert response.parsed.model_dump() == reply


async def test_per_call_list_of_places_refuses_a_place_not_offered() -> None:
    llm = _llm(_reply(_message('{"reason": "r", "reply_to": [{"place": 1, "point": "p"}]}')))
    with pytest.raises(LLMError, match="SpeakDecision"):
        await llm.generate(_request(schema=_decision_schema([0])))


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


@pytest.mark.parametrize(
    ("body", "schema"),
    [
        (_message("No.", stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}), None),
        (_message('{"name": "buil', stop_reason="max_tokens"), Answer),
        (_message('{"ok": "maybe"}'), Answer),
    ],
    ids=["refusal", "cut off", "schema mismatch"],
)
async def test_a_failure_the_provider_answered_carries_the_answer(body: dict[str, Any], schema: Any) -> None:
    # Sonnet was asked for, Haiku answered: 1,000 input at $0.10/MTok + 200 output at $0.50/MTok
    with pytest.raises(LLMError) as raised:
        await _llm(_reply(body)).generate(_request(ModelTier.STRONG, schema=schema))

    error = raised.value
    assert error.model == "claude-haiku-5-5"
    assert error.usage == Usage(input_tokens=1_000, output_tokens=200)
    assert error.cost_usd == pytest.approx(0.0002)
    assert error.text == body["content"][0]["text"]


async def test_an_api_error_carries_only_the_requested_model() -> None:
    with pytest.raises(LLMError, match="server error") as raised:
        await _llm(_error(500)).generate(_request(ModelTier.STRONG))

    error = raised.value
    assert (error.model, error.usage, error.cost_usd, error.text) == ("claude-sonnet-5-5", None, 0.0, None)


async def test_a_timeout_carries_only_the_requested_model() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError, match="timed out") as raised:
        await _llm(handler).generate(_request())

    assert (raised.value.model, raised.value.usage, raised.value.cost_usd) == ("claude-haiku-5-5", None, 0.0)


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


async def test_cost_uses_the_price_of_the_model_that_answered(caplog: pytest.LogCaptureFixture) -> None:
    # Sonnet was asked for, Haiku answered: 1,000 input at $0.10/MTok + 200 output at $0.50/MTok
    response = await _llm(_reply(_message("hi", model="claude-haiku-5-5"))).generate(_request(ModelTier.STRONG))

    assert response.model == "claude-haiku-5-5"
    assert response.cost_usd == pytest.approx(0.0002)
    assert "priced by the requested model" not in caplog.text


async def test_an_answering_model_without_a_price_is_priced_as_requested_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm(_reply(_message("hi", model="claude-sonnet-5-5-20261001")))

    with caplog.at_level(logging.WARNING, logger="techdebate"):
        first = await llm.generate(_request(ModelTier.STRONG))
        second = await llm.generate(_request(ModelTier.STRONG))

    assert first.model == "claude-sonnet-5-5-20261001"
    assert first.cost_usd == second.cost_usd == pytest.approx(0.004)  # Sonnet's price
    line = "llm priced by the requested model requested=claude-sonnet-5-5 answered=claude-sonnet-5-5-20261001"
    assert caplog.messages == [line, line]  # on every such call


def test_cost_usd() -> None:
    price = ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50)
    assert cost_usd(Usage(input_tokens=2_000_000, output_tokens=1_000_000), price) == pytest.approx(0.70)


def test_model_without_price_fails_at_construction() -> None:
    with pytest.raises(LLMError, match="claude-unknown"):
        _llm(_reply(_message("x")), strong_model="claude-unknown")


def test_satisfies_protocol() -> None:
    client: LLMClient = _llm(_reply(_message("x")))
    assert client is not None
