import logging

import pytest

from techdebate.errors import LLMError
from techdebate.llm import CostMeter, FakeLLM, LLMRequest, LLMResponse, Message, ModelTier


class _PricedLLM(FakeLLM):
    """Each call costs a set amount, by prompt."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await super().generate(request)
        prices = {"decide.v1": 0.01, "developer.v1": 0.25}
        return response.model_copy(update={"cost_usd": prices[request.prompt_id or ""]})


def _request(prompt_id: str) -> LLMRequest:
    return LLMRequest(
        tier=ModelTier.CHEAP,
        system="s",
        messages=[Message(role="user", content="hi")],
        max_tokens=10,
        prompt_id=prompt_id,
    )


def _llm() -> _PricedLLM:
    return _PricedLLM(rules=[(lambda _: True, "ok")])


async def test_the_meter_passes_each_call_on_and_returns_its_response() -> None:
    llm = _llm()
    meter = CostMeter(llm)

    response = await meter.generate(_request("decide.v1"))

    assert response.text == "ok"
    assert [request.prompt_id for request in llm.calls] == ["decide.v1"]


async def test_the_meter_adds_up_the_cost_of_every_call() -> None:
    meter = CostMeter(_llm())
    assert meter.total_usd == 0.0

    for prompt_id in ("decide.v1", "decide.v1", "developer.v1"):
        await meter.generate(_request(prompt_id))

    assert meter.total_usd == pytest.approx(0.27)


async def test_a_failed_call_adds_nothing_and_goes_up() -> None:
    meter = CostMeter(FakeLLM())

    with pytest.raises(LLMError, match="no scripted reply"):
        await meter.generate(_request("decide.v1"))

    assert meter.total_usd == 0.0


async def test_each_call_is_logged_at_debug_with_its_numbers(caplog: pytest.LogCaptureFixture) -> None:
    meter = CostMeter(_llm())

    with caplog.at_level(logging.DEBUG, logger="techdebate.llm.meter"):
        await meter.generate(_request("developer.v1"))

    [record] = caplog.records
    assert record.levelno == logging.DEBUG
    message = record.getMessage()
    assert message.startswith("llm call prompt=developer.v1 model=fake input_tokens=0 output_tokens=0 cost=0.250000 ")
    assert "duration_s=" in message
