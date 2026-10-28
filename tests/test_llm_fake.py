from typing import Literal

import pytest
from pydantic import BaseModel, create_model

from techdebate.errors import LLMError
from techdebate.llm import FakeLLM, LLMClient, LLMRequest, Message, ModelTier
from techdebate.llm.fake import Reply


class Answer(BaseModel):
    name: str
    ok: bool
    reason: str


class Choice(BaseModel):
    """A fixed model a test can script, standing in for a schema built per call."""

    reply_to: int | None
    reason: str


# Built per call, the way a schema that only allows the valid choices is.
Pick = create_model("Pick", reply_to=(Literal[1, 2] | None, ...), reason=(str, ...))


def _request(prompt: str, tier: ModelTier = ModelTier.CHEAP) -> LLMRequest:
    return LLMRequest(tier=tier, system="sys", messages=[Message(role="user", content=prompt)], max_tokens=100)


async def _generate_pick(reply: Reply) -> BaseModel | None:
    llm = FakeLLM(rules=[(lambda _: True, reply)])
    response = await llm.generate(_request("Rust").model_copy(update={"output_schema": Pick}))
    return response.parsed


async def test_reply_keyed_by_prompt() -> None:
    llm = FakeLLM(by_prompt={"hello": "hi there"})
    response = await llm.generate(_request("hello"))
    assert response.text == "hi there"
    assert response.parsed is None
    assert response.model == "fake"
    assert response.cost_usd == 0.0


async def test_prompt_id_is_copied_to_response() -> None:
    llm = FakeLLM(by_prompt={"hello": "hi there"})
    assert (await llm.generate(_request("hello"))).prompt_id is None
    request = _request("hello").model_copy(update={"prompt_id": "greet.v1"})
    assert (await llm.generate(request)).prompt_id == "greet.v1"


async def test_reply_matched_by_function() -> None:
    llm = FakeLLM(rules=[(lambda r: r.tier == ModelTier.STRONG, "strong answer")])
    response = await llm.generate(_request("anything", tier=ModelTier.STRONG))
    assert response.text == "strong answer"


async def test_model_reply_sets_parsed_and_text() -> None:
    answer = Answer(name="builder", ok=True, reason="Widely used")
    llm = FakeLLM(rules=[(lambda _: True, answer)])
    response = await llm.generate(_request("Rust").model_copy(update={"output_schema": Answer}))
    assert response.parsed == answer
    assert Answer.model_validate_json(response.text) == answer


async def test_model_reply_is_validated_into_per_call_schema() -> None:
    parsed = await _generate_pick(Choice(reply_to=2, reason="Disagree"))
    assert isinstance(parsed, Pick)
    assert parsed.model_dump() == {"reply_to": 2, "reason": "Disagree"}


async def test_text_reply_is_parsed_into_schema() -> None:
    parsed = await _generate_pick('{"reply_to": null, "reason": "New point"}')
    assert isinstance(parsed, Pick)
    assert parsed.model_dump() == {"reply_to": None, "reason": "New point"}


@pytest.mark.parametrize("reply", [Choice(reply_to=3, reason="No such turn"), "not json"])
async def test_reply_not_matching_schema_raises(reply: Reply) -> None:
    with pytest.raises(LLMError, match="Pick"):
        await _generate_pick(reply)


async def test_model_reply_without_schema_is_text_only() -> None:
    answer = Answer(name="builder", ok=True, reason="Widely used")
    response = await FakeLLM(rules=[(lambda _: True, answer)]).generate(_request("Rust"))
    assert response.parsed is None
    assert Answer.model_validate_json(response.text) == answer


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
