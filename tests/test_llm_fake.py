import pytest

from techdebate.errors import LLMError
from techdebate.llm import FakeLLM, LLMClient, LLMRequest, Message, ModelTier
from techdebate.models import RelevanceDecision


def _request(prompt: str, tier: ModelTier = ModelTier.CHEAP) -> LLMRequest:
    return LLMRequest(tier=tier, system="sys", messages=[Message(role="user", content=prompt)], max_tokens=100)


async def test_reply_keyed_by_prompt() -> None:
    llm = FakeLLM(by_prompt={"hello": "hi there"})
    response = await llm.generate(_request("hello"))
    assert response.text == "hi there"
    assert response.parsed is None
    assert response.model == "fake"
    assert response.cost_usd == 0.0


async def test_reply_matched_by_function() -> None:
    llm = FakeLLM(rules=[(lambda r: r.tier == ModelTier.STRONG, "strong answer")])
    response = await llm.generate(_request("anything", tier=ModelTier.STRONG))
    assert response.text == "strong answer"


async def test_model_reply_sets_parsed_and_text() -> None:
    decision = RelevanceDecision(analyst="builder", join=True, reason="Widely used")
    llm = FakeLLM(rules=[(lambda _: True, decision)])
    response = await llm.generate(_request("Rust").model_copy(update={"output_schema": RelevanceDecision}))
    assert response.parsed == decision
    assert RelevanceDecision.model_validate_json(response.text) == decision


async def test_calls_are_recorded_in_order() -> None:
    llm = FakeLLM(rules=[(lambda _: True, "ok")])
    first, second = _request("one"), _request("two")
    await llm.generate(first)
    await llm.generate(second)
    assert llm.calls == [first, second]


async def test_unmatched_request_raises() -> None:
    llm = FakeLLM(by_prompt={"hello": "hi"})
    with pytest.raises(LLMError, match="goodbye"):
        await llm.generate(_request("goodbye"))


def test_fake_satisfies_protocol() -> None:
    client: LLMClient = FakeLLM()
    assert client is not None
