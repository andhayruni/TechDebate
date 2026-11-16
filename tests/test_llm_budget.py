from dataclasses import dataclass

import pytest

from techdebate.config import Settings
from techdebate.errors import BudgetExceeded
from techdebate.llm import (
    Budget,
    BudgetScope,
    FakeLLM,
    LLMRequest,
    LLMResponse,
    Message,
    ModelTier,
    RecordingLLM,
    post_scopes,
)
from techdebate.models import Message as Post
from techdebate.models import User, UserKind
from techdebate.store import Store


class _PricedLLM(FakeLLM):
    """Every call costs $0.25."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await super().generate(request)
        return response.model_copy(update={"cost_usd": 0.25})


@dataclass(frozen=True)
class _SpentScope(BudgetScope):
    """A scope with a set total, to check the order of the scopes."""

    spent: float = 0.0

    async def spent_usd(self, store: Store, agent: User, post_id: str) -> float:
        return self.spent


def _request() -> LLMRequest:
    return LLMRequest(
        tier=ModelTier.STRONG,
        system="s",
        messages=[Message(role="user", content="hi")],
        max_tokens=10,
        prompt_id="developer.v1",
    )


async def _users_and_post(store: Store) -> tuple[User, User, Post]:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    scientist = User(username="scientist", kind=UserKind.AGENT)
    post = Post(author_id=me.id, text="Is Rust growing?")
    for user in (me, developer, scientist):
        await store.add_user(user)
    await store.add_message(post)
    return developer, scientist, post


def _budget(llm: FakeLLM, store: Store, agent: User, post: Post, limit_usd: float) -> Budget:
    recording = RecordingLLM(llm, store, agent=agent, post_id=post.id)
    return Budget(recording, store, [BudgetScope(name="post", limit_usd=limit_usd)], agent=agent, post_id=post.id)


async def test_calls_are_passed_on_until_the_post_has_spent_its_budget(store: Store) -> None:
    developer, _, post = await _users_and_post(store)
    llm = _PricedLLM(rules=[(lambda _: True, "ok")])
    budget = _budget(llm, store, developer, post, limit_usd=0.50)

    await budget.generate(_request())
    await budget.generate(_request())  # $0.25 spent, under the budget
    with pytest.raises(BudgetExceeded) as raised:
        await budget.generate(_request())  # $0.50 spent: the budget is used up

    assert len(llm.calls) == 2  # the refused call never reached the provider
    assert len(await store.list_llm_calls(post.id)) == 2  # and nothing was stored for it
    error = raised.value
    assert (error.scope, error.spent_usd, error.limit_usd) == ("post", pytest.approx(0.50), 0.50)
    assert str(error) == "the post has spent $0.5000 of its $0.50 budget"


async def test_the_post_budget_counts_the_calls_of_every_agent_on_the_post(store: Store) -> None:
    developer, scientist, post = await _users_and_post(store)
    llm = _PricedLLM(rules=[(lambda _: True, "ok")])
    await _budget(llm, store, developer, post, limit_usd=0.30).generate(_request())

    with pytest.raises(BudgetExceeded):
        await _budget(llm, store, scientist, post, limit_usd=0.25).check()
    await _budget(llm, store, scientist, post, limit_usd=0.30).check()  # $0.25 is below $0.30


async def test_a_call_on_another_post_does_not_count(store: Store) -> None:
    developer, _, post = await _users_and_post(store)
    other = Post(author_id=post.author_id, text="Is Go growing?")
    await store.add_message(other)
    llm = _PricedLLM(rules=[(lambda _: True, "ok")])
    await _budget(llm, store, developer, other, limit_usd=1.0).generate(_request())

    await _budget(llm, store, developer, post, limit_usd=0.25).check()


async def test_the_first_scope_used_up_is_the_one_named(store: Store) -> None:
    developer, _, post = await _users_and_post(store)
    scopes = [
        _SpentScope(name="post", limit_usd=0.50, spent=0.10),
        _SpentScope(name="agent", limit_usd=1.00, spent=1.00),
        _SpentScope(name="day", limit_usd=5.00, spent=6.00),
    ]
    budget = Budget(FakeLLM(), store, scopes, agent=developer, post_id=post.id)

    with pytest.raises(BudgetExceeded, match=r"^the agent has spent \$1\.0000 of its \$1\.00 budget$"):
        await budget.check()


def test_the_post_scope_takes_its_limit_from_the_settings() -> None:
    settings = Settings(_env_file=None, post_budget_usd=0.02)  # pyright: ignore[reportCallIssue]

    assert post_scopes(settings) == [BudgetScope(name="post", limit_usd=0.02)]
