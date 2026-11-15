import logging
from datetime import UTC

import pytest

from techdebate.errors import LLMError
from techdebate.llm import FakeLLM, LLMRequest, LLMResponse, Message, ModelTier, RecordingLLM, Usage
from techdebate.models import Message as Post
from techdebate.models import User, UserKind
from techdebate.store import Store


class _PricedLLM(FakeLLM):
    """Every call answers with set numbers, from a model other than the fake's own name."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await super().generate(request)
        usage = Usage(input_tokens=1234, output_tokens=56)
        return response.model_copy(update={"usage": usage, "cost_usd": 0.25, "model": "claude-sonnet-5-5"})


def _request(prompt_id: str | None = "developer.v1") -> LLMRequest:
    return LLMRequest(
        tier=ModelTier.STRONG,
        system="s",
        messages=[Message(role="user", content="hi")],
        max_tokens=10,
        prompt_id=prompt_id,
    )


async def _agent_and_post(store: Store) -> tuple[User, Post]:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    post = Post(author_id=me.id, text="Is Rust growing?")
    await store.add_user(me)
    await store.add_user(developer)
    await store.add_message(post)
    return developer, post


async def test_a_call_is_passed_on_and_stored_with_its_agent_post_and_numbers(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    llm = _PricedLLM(rules=[(lambda _: True, "ok")])
    recording = RecordingLLM(llm, store, agent=developer, post_id=post.id)

    response = await recording.generate(_request())

    assert response.text == "ok"
    assert [request.prompt_id for request in llm.calls] == ["developer.v1"]
    [call] = await store.list_llm_calls(post.id)
    assert call.agent_id == developer.id
    assert call.post_id == post.id
    assert call.prompt_id == "developer.v1"
    assert call.model == "claude-sonnet-5-5"
    assert (call.input_tokens, call.output_tokens) == (1234, 56)
    assert call.cost_usd == 0.25
    assert call.duration_s >= 0
    assert call.created_at.tzinfo is UTC


async def test_every_call_gets_its_own_row(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    recording = RecordingLLM(FakeLLM(rules=[(lambda _: True, "ok")]), store, agent=developer, post_id=post.id)

    for prompt_id in ("research.v1", "decide.v1", "developer.v1"):
        await recording.generate(_request(prompt_id))

    calls = await store.list_llm_calls(post.id)
    assert [call.prompt_id for call in calls] == ["research.v1", "decide.v1", "developer.v1"]


async def test_a_failed_call_stores_nothing_and_goes_up(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    recording = RecordingLLM(FakeLLM(), store, agent=developer, post_id=post.id)

    with pytest.raises(LLMError, match="no scripted reply"):
        await recording.generate(_request())

    assert await store.list_llm_calls(post.id) == []


async def test_a_request_without_a_prompt_id_is_refused_before_the_call(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    llm = FakeLLM(rules=[(lambda _: True, "ok")])
    recording = RecordingLLM(llm, store, agent=developer, post_id=post.id)

    with pytest.raises(ValueError, match="needs a prompt ID"):
        await recording.generate(_request(prompt_id=None))

    assert llm.calls == []
    assert await store.list_llm_calls(post.id) == []


async def test_each_call_is_logged_at_debug_with_its_agent_post_and_numbers(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    developer, post = await _agent_and_post(store)
    recording = RecordingLLM(_PricedLLM(rules=[(lambda _: True, "ok")]), store, agent=developer, post_id=post.id)

    with caplog.at_level(logging.DEBUG, logger="techdebate.llm.recording"):
        await recording.generate(_request())

    [record] = caplog.records
    assert record.levelno == logging.DEBUG
    message = record.getMessage()
    assert message.startswith(
        f"llm call agent=developer post_id={post.id} prompt=developer.v1 model=claude-sonnet-5-5 "
        "input_tokens=1234 output_tokens=56 cost=0.250000 "
    )
    assert "duration_s=" in message
