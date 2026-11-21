from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from techdebate.config import Settings
from techdebate.errors import BudgetExceeded, LLMError, usd_limit
from techdebate.llm import (
    Budget,
    BudgetScope,
    FakeLLM,
    LLMRequest,
    LLMResponse,
    Message,
    ModelTier,
    RecordingLLM,
    Usage,
    budget_scopes,
    end_of_day,
    start_of_day,
)
from techdebate.models import AgentVisit, LLMCall, User, UserKind
from techdebate.models import Message as Post
from techdebate.store import Store


class _PricedLLM(FakeLLM):
    """Every call costs $0.25."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await super().generate(request)
        return response.model_copy(update={"cost_usd": 0.25})


class _RefusingLLM(FakeLLM):
    """Every call is refused after the provider answered, at $0.25."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMError("m refused the request", model="m", usage=Usage(input_tokens=1, output_tokens=1), cost_usd=0.25)


@dataclass(frozen=True)
class _SpentScope(BudgetScope):
    """A scope with a set total, to check the order of the scopes."""

    spent: float = 0.0

    async def spent_usd(
        self, store: Store, *, agent_id: str | None = None, post_id: str | None = None, now: datetime | None = None
    ) -> float:
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


async def _budget(llm: FakeLLM, store: Store, agent: User, post: Post, limit_usd: float) -> Budget:
    visit = AgentVisit(agent_id=agent.id, post_id=post.id)
    await store.add_visit(visit)
    recording = RecordingLLM(llm, store, agent=agent, post_id=post.id, visit_id=visit.id)
    scope = BudgetScope(name="post", limit_usd=limit_usd, by_post=True)
    return Budget(recording, store, [scope], agent=agent, post_id=post.id)


async def _stored_call(store: Store, agent: User, post: Post, cost_usd: float, created_at: datetime) -> None:
    """A call stored as if it had been made at `created_at`."""
    await store.add_llm_call(
        LLMCall(
            agent_id=agent.id,
            post_id=post.id,
            prompt_id="developer.v1",
            model="m",
            input_tokens=1,
            output_tokens=1,
            cost_usd=cost_usd,
            duration_s=0.1,
            created_at=created_at,
        )
    )


async def test_calls_are_passed_on_until_the_post_has_spent_its_budget(store: Store) -> None:
    developer, _, post = await _users_and_post(store)
    llm = _PricedLLM(rules=[(lambda _: True, "ok")])
    budget = await _budget(llm, store, developer, post, limit_usd=0.50)

    await budget.generate(_request())
    await budget.generate(_request())  # $0.25 spent, under the budget
    with pytest.raises(BudgetExceeded) as raised:
        await budget.generate(_request())  # $0.50 spent: the budget is used up

    assert len(llm.calls) == 2  # the refused call never reached the provider
    assert len(await store.list_llm_calls(post.id)) == 2  # and nothing was stored for it
    error = raised.value
    assert (error.scope, error.spent_usd, error.limit_usd) == ("post", pytest.approx(0.50), 0.50)
    assert str(error) == "the post has spent $0.5000 of its $0.50 budget"


async def test_a_failed_call_that_cost_something_counts(store: Store) -> None:
    developer, _, post = await _users_and_post(store)
    budget = await _budget(_RefusingLLM(), store, developer, post, limit_usd=0.25)

    with pytest.raises(LLMError, match="refused"):
        await budget.generate(_request())  # paid for, though it failed
    with pytest.raises(BudgetExceeded):
        await budget.check()


async def test_the_post_budget_counts_the_calls_of_every_agent_on_the_post(store: Store) -> None:
    developer, scientist, post = await _users_and_post(store)
    llm = _PricedLLM(rules=[(lambda _: True, "ok")])
    await (await _budget(llm, store, developer, post, limit_usd=0.30)).generate(_request())

    with pytest.raises(BudgetExceeded):
        await (await _budget(llm, store, scientist, post, limit_usd=0.25)).check()
    await (await _budget(llm, store, scientist, post, limit_usd=0.30)).check()  # $0.25 is below $0.30


async def test_a_call_on_another_post_does_not_count(store: Store) -> None:
    developer, _, post = await _users_and_post(store)
    other = Post(author_id=post.author_id, text="Is Go growing?")
    await store.add_message(other)
    llm = _PricedLLM(rules=[(lambda _: True, "ok")])
    await (await _budget(llm, store, developer, other, limit_usd=1.0)).generate(_request())

    await (await _budget(llm, store, developer, post, limit_usd=0.25)).check()


async def test_the_first_scope_used_up_is_the_one_named(store: Store) -> None:
    developer, _, post = await _users_and_post(store)
    scopes = [
        _SpentScope(name="post", limit_usd=0.50, spent=0.10),
        _SpentScope(name="agent", limit_usd=1.00, spent=1.00),
        _SpentScope(name="day", limit_usd=5.00, spent=6.00),
    ]
    budget = Budget(FakeLLM(), store, scopes, agent=developer, post_id=post.id)

    with pytest.raises(BudgetExceeded, match=r"^the agent has spent \$1\.0000 of its \$1\.00 budget for today$"):
        await budget.check()


def test_the_scopes_take_their_limits_from_the_settings_in_the_order_post_agent_day() -> None:
    limits = {"post_budget_usd": 0.02, "agent_budget_usd": 0.5, "daily_budget_usd": 1.0}
    settings = Settings(_env_file=None, **limits)  # pyright: ignore[reportCallIssue]

    assert budget_scopes(settings) == [
        BudgetScope(name="post", limit_usd=0.02, by_post=True),
        BudgetScope(name="agent", limit_usd=0.5, by_agent=True, today=True),
        BudgetScope(name="day", limit_usd=1.0, today=True),
    ]


def test_the_default_budgets() -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    assert (settings.post_budget_usd, settings.agent_budget_usd, settings.daily_budget_usd) == (0.50, 2.00, 5.00)


@pytest.mark.parametrize("name", ["post_budget_usd", "agent_budget_usd", "daily_budget_usd"])
def test_a_budget_below_zero_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Settings(_env_file=None, **{name: -0.01})  # pyright: ignore[reportCallIssue]


async def test_the_agent_scope_counts_the_agents_calls_of_today_on_every_post(store: Store) -> None:
    developer, scientist, post = await _users_and_post(store)
    other = Post(author_id=post.author_id, text="Is Go growing?")
    await store.add_message(other)
    now = datetime.now(UTC)
    await _stored_call(store, developer, post, 0.10, now)
    await _stored_call(store, developer, other, 0.20, now)  # another post counts
    await _stored_call(store, scientist, post, 0.40, now)  # another agent does not
    await _stored_call(store, developer, post, 0.80, start_of_day(now) - timedelta(seconds=1))  # nor yesterday
    scope = budget_scopes(Settings(_env_file=None))[1]  # pyright: ignore[reportCallIssue]

    assert await scope.spent_usd(store, agent_id=developer.id, now=now) == pytest.approx(0.30)


async def test_the_day_scope_counts_every_agents_calls_of_today(store: Store) -> None:
    developer, scientist, post = await _users_and_post(store)
    now = datetime.now(UTC)
    await _stored_call(store, developer, post, 0.10, now)
    await _stored_call(store, scientist, post, 0.40, start_of_day(now))  # the first moment of today counts
    await _stored_call(store, developer, post, 0.80, start_of_day(now) - timedelta(seconds=1))
    scope = budget_scopes(Settings(_env_file=None))[2]  # pyright: ignore[reportCallIssue]

    assert await scope.spent_usd(store, now=now) == pytest.approx(0.50)


async def test_an_agent_out_of_its_budget_today_is_stopped_on_every_post_and_another_agent_is_not(
    store: Store,
) -> None:
    developer, scientist, post = await _users_and_post(store)
    other = Post(author_id=post.author_id, text="Is Go growing?")
    await store.add_message(other)
    await _stored_call(store, developer, other, 0.50, datetime.now(UTC))
    scopes = budget_scopes(Settings(_env_file=None, agent_budget_usd=0.50))  # pyright: ignore[reportCallIssue]

    with pytest.raises(BudgetExceeded, match=r"^the agent has spent \$0\.5000 of its \$0\.50 budget for today$"):
        await Budget(FakeLLM(), store, scopes, agent=developer, post_id=post.id).check()
    await Budget(FakeLLM(), store, scopes, agent=scientist, post_id=post.id).check()


async def test_the_agents_out_of_the_daily_budget_are_all_stopped_until_the_next_day(store: Store) -> None:
    developer, scientist, post = await _users_and_post(store)
    other = Post(author_id=post.author_id, text="Is Go growing?")
    await store.add_message(other)
    settings = Settings(_env_file=None, daily_budget_usd=1.00)  # pyright: ignore[reportCallIssue]
    await _stored_call(store, developer, post, 0.60, datetime.now(UTC))
    await _stored_call(store, scientist, post, 0.40, datetime.now(UTC))

    for agent in (developer, scientist):
        budget = Budget(FakeLLM(), store, budget_scopes(settings), agent=agent, post_id=other.id)
        with pytest.raises(
            BudgetExceeded, match=r"^the agents have spent \$1\.0000 of their \$1\.00 budget for today$"
        ):
            await budget.check()


async def test_calls_of_yesterday_do_not_count_against_today(store: Store) -> None:
    developer, _, post = await _users_and_post(store)
    yesterday = datetime.now(UTC) - timedelta(days=1)
    await _stored_call(store, developer, post, 9.00, yesterday)
    settings = Settings(_env_file=None, post_budget_usd=10.0)  # pyright: ignore[reportCallIssue]

    await Budget(FakeLLM(), store, budget_scopes(settings), agent=developer, post_id=post.id).check()


async def test_a_scope_that_counts_a_post_or_an_agent_needs_its_id(store: Store) -> None:
    post, agent, _ = budget_scopes(Settings(_env_file=None))  # pyright: ignore[reportCallIssue]

    with pytest.raises(ValueError, match="post"):
        await post.spent_usd(store, agent_id="a")
    with pytest.raises(ValueError, match="agent"):
        await agent.spent_usd(store, post_id="p")


def test_the_day_is_the_utc_day() -> None:
    late_in_new_york = datetime(2026, 11, 16, 23, 30, tzinfo=ZoneInfo("America/New_York"))  # 04:30 on the 17th in UTC

    assert start_of_day(late_in_new_york) == datetime(2026, 11, 17, tzinfo=UTC)
    assert end_of_day(late_in_new_york) == datetime(2026, 11, 18, tzinfo=UTC)


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        ("post", "the post has spent $0.5021 of its $0.50 budget"),
        ("agent", "the agent has spent $0.5021 of its $0.50 budget for today"),
        ("day", "the agents have spent $0.5021 of their $0.50 budget for today"),
        ("user", "the user has spent $0.5021 of its $0.50 budget"),
    ],
)
def test_the_message_names_the_scope(scope: str, message: str) -> None:
    assert str(BudgetExceeded(scope, 0.5021, 0.50)) == message


@pytest.mark.parametrize(
    ("limit", "shown"), [(0.5, "$0.50"), (2, "$2.00"), (0.015, "$0.015"), (0.075, "$0.075"), (0.0001, "$0.0001")]
)
def test_a_limit_is_shown_with_cents_and_more_digits_only_when_it_has_them(limit: float, shown: str) -> None:
    assert usd_limit(limit) == shown
    assert str(BudgetExceeded("post", 0.02, limit)).endswith(f"of its {shown} budget")
