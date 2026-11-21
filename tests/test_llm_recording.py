import logging
from datetime import UTC

import pytest

from techdebate.errors import LLMError
from techdebate.llm import FakeLLM, LLMClient, LLMRequest, LLMResponse, Message, ModelTier, RecordingLLM, Usage
from techdebate.models import AgentVisit, ResearchPlan, User, UserKind
from techdebate.models import Message as Post
from techdebate.store import Store


class _PricedLLM(FakeLLM):
    """Every call answers with set numbers, from a model other than the fake's own name."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await super().generate(request)
        usage = Usage(input_tokens=1234, output_tokens=56)
        return response.model_copy(update={"usage": usage, "cost_usd": 0.25, "model": "claude-sonnet-5-5"})


class _FailingLLM:
    """Every call raises the error given."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise self._error


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


async def _recording(llm: LLMClient, store: Store, agent: User, post: Post) -> tuple[RecordingLLM, AgentVisit]:
    """A recording for a visit of the agent to the post, stored as a visit is."""
    visit = AgentVisit(agent_id=agent.id, post_id=post.id)
    await store.add_visit(visit)
    return RecordingLLM(llm, store, agent=agent, post_id=post.id, visit_id=visit.id), visit


async def test_a_call_is_passed_on_and_stored_with_its_agent_post_and_numbers(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    llm = _PricedLLM(rules=[(lambda _: True, "ok")])
    recording, visit = await _recording(llm, store, developer, post)

    response = await recording.generate(_request())

    assert response.text == "ok"
    assert [request.prompt_id for request in llm.calls] == ["developer.v1"]
    [call] = await store.list_llm_calls(post.id)
    assert call.agent_id == developer.id
    assert call.post_id == post.id
    assert call.visit_id == visit.id
    assert call.prompt_id == "developer.v1"
    assert call.model == "claude-sonnet-5-5"
    assert (call.input_tokens, call.output_tokens) == (1234, 56)
    assert call.cost_usd == 0.25
    assert call.duration_s >= 0
    assert (call.output, call.error) == ("ok", None)
    assert call.created_at.tzinfo is UTC
    assert recording.total_usd == 0.25


async def test_every_call_gets_its_own_row(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    recording, _ = await _recording(FakeLLM(rules=[(lambda _: True, "ok")]), store, developer, post)

    for prompt_id in ("research.v1", "decide.v1", "developer.v1"):
        await recording.generate(_request(prompt_id))

    calls = await store.list_llm_calls(post.id)
    assert [call.prompt_id for call in calls] == ["research.v1", "decide.v1", "developer.v1"]


async def test_a_structured_answer_is_stored_as_its_json(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    plan = ResearchPlan(speak=False, reason="Not my area.")
    recording, _ = await _recording(FakeLLM(rules=[(lambda _: True, plan)]), store, developer, post)

    await recording.generate(_request("research.v1").model_copy(update={"output_schema": ResearchPlan}))

    [call] = await store.list_llm_calls(post.id)
    assert call.output is not None
    assert ResearchPlan.model_validate_json(call.output) == plan


async def test_a_failed_call_the_provider_answered_is_stored_with_its_answer_and_goes_up(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    usage = Usage(input_tokens=900, output_tokens=40)
    error = LLMError("m refused the request", model="claude-haiku-5-5", usage=usage, cost_usd=0.002, text="No.")
    recording, visit = await _recording(_FailingLLM(error), store, developer, post)

    with pytest.raises(LLMError) as raised:
        await recording.generate(_request())

    assert raised.value is error
    [call] = await store.list_llm_calls(post.id)
    assert (call.visit_id, call.prompt_id, call.model) == (visit.id, "developer.v1", "claude-haiku-5-5")
    assert (call.input_tokens, call.output_tokens, call.cost_usd) == (900, 40, 0.002)
    assert (call.output, call.error) == ("No.", "m refused the request")
    assert call.duration_s >= 0
    assert recording.total_usd == 0.002


async def test_a_failed_call_with_no_answer_is_stored_with_no_tokens_or_cost(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    recording, _ = await _recording(
        _FailingLLM(LLMError("Anthropic request timed out", model="claude-sonnet-5-5")), store, developer, post
    )

    with pytest.raises(LLMError, match="timed out"):
        await recording.generate(_request())

    [call] = await store.list_llm_calls(post.id)
    assert (call.model, call.input_tokens, call.output_tokens, call.cost_usd) == ("claude-sonnet-5-5", 0, 0, 0.0)
    assert (call.output, call.error) == (None, "Anthropic request timed out")


async def test_a_failure_that_names_no_model_is_stored_with_an_unknown_model(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    recording, _ = await _recording(_FailingLLM(LLMError("down")), store, developer, post)

    with pytest.raises(LLMError):
        await recording.generate(_request())

    [call] = await store.list_llm_calls(post.id)
    assert (call.model, call.error) == ("unknown", "down")


async def test_a_failure_of_the_fake_is_stored_like_a_real_one(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    recording, _ = await _recording(FakeLLM(), store, developer, post)

    with pytest.raises(LLMError, match="no scripted reply"):
        await recording.generate(_request())

    [call] = await store.list_llm_calls(post.id)
    assert call.model == "fake"
    assert call.error is not None and "no scripted reply" in call.error


async def test_an_error_other_than_llm_error_stores_nothing_and_goes_up(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    recording, _ = await _recording(_FailingLLM(RuntimeError("bug")), store, developer, post)

    with pytest.raises(RuntimeError, match="bug"):
        await recording.generate(_request())

    assert await store.list_llm_calls(post.id) == []


async def test_a_request_without_a_prompt_id_is_refused_before_the_call(store: Store) -> None:
    developer, post = await _agent_and_post(store)
    llm = FakeLLM(rules=[(lambda _: True, "ok")])
    recording, _ = await _recording(llm, store, developer, post)

    with pytest.raises(ValueError, match="needs a prompt ID"):
        await recording.generate(_request(prompt_id=None))

    assert llm.calls == []
    assert await store.list_llm_calls(post.id) == []


async def test_each_call_is_logged_at_debug_with_its_agent_post_and_numbers(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    developer, post = await _agent_and_post(store)
    recording, _ = await _recording(_PricedLLM(rules=[(lambda _: True, "ok")]), store, developer, post)

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
    assert "error=" not in message


async def test_a_failed_call_is_logged_with_its_error(store: Store, caplog: pytest.LogCaptureFixture) -> None:
    developer, post = await _agent_and_post(store)
    recording, _ = await _recording(_FailingLLM(LLMError("down", model="m")), store, developer, post)

    with caplog.at_level(logging.DEBUG, logger="techdebate.llm.recording"), pytest.raises(LLMError):
        await recording.generate(_request())

    [record] = caplog.records
    assert record.getMessage().endswith(" error=down")
