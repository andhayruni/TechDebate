"""A client that refuses a call once a budget is used up, for one agent and one post. It wraps the recording, so a
refused call reaches neither the provider nor the stored calls.

Three budgets cover an agent's calls: the post's, the agent's for today, and all agents' for today. The day is the UTC
day, and its start is worked out again on every check, so the agents running past midnight count the new day at once."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from techdebate.config import Settings
from techdebate.errors import BudgetExceeded
from techdebate.llm.base import LLMClient, LLMRequest, LLMResponse
from techdebate.models import User
from techdebate.store.base import Store


def start_of_day(now: datetime) -> datetime:
    """The start of the UTC day that `now` is in, which the daily budgets count from."""
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_day(now: datetime) -> datetime:
    """When the daily budgets reset."""
    return start_of_day(now) + timedelta(days=1)


@dataclass(frozen=True)
class BudgetScope:
    """What one budget covers: the calls on the post, those of the agent, and only those of today, for each flag set.
    With none set, it counts every call."""

    name: str  # how BudgetExceeded names it: "post", "agent" or "day"
    limit_usd: float
    by_post: bool = False
    by_agent: bool = False
    today: bool = False

    async def spent_usd(
        self, store: Store, *, agent_id: str | None = None, post_id: str | None = None, now: datetime | None = None
    ) -> float:
        """What the scope has spent. The agent and the post are needed only by a scope that counts them."""
        if self.by_post and post_id is None:
            raise ValueError(f"the {self.name} budget counts the calls on a post: give its ID")
        if self.by_agent and agent_id is None:
            raise ValueError(f"the {self.name} budget counts the calls of an agent: give its ID")
        return await store.spent_usd(
            post_id=post_id if self.by_post else None,
            agent_id=agent_id if self.by_agent else None,
            since=start_of_day(now or datetime.now(UTC)) if self.today else None,
        )

    def used_up(self, spent_usd: float) -> bool:
        return spent_usd >= self.limit_usd


def budget_scopes(settings: Settings) -> list[BudgetScope]:
    """The budgets an agent's calls on a post are checked against, in the order they are named: a post used up for
    good comes before a day that resets."""
    return [
        BudgetScope(name="post", limit_usd=settings.post_budget_usd, by_post=True),
        BudgetScope(name="agent", limit_usd=settings.agent_budget_usd, by_agent=True, today=True),
        BudgetScope(name="day", limit_usd=settings.daily_budget_usd, today=True),
    ]


class Budget:
    def __init__(
        self, llm: LLMClient, store: Store, scopes: Sequence[BudgetScope], *, agent: User, post_id: str
    ) -> None:
        self._llm = llm
        self._store = store
        self._scopes = tuple(scopes)
        self._agent = agent
        self._post_id = post_id

    async def check(self) -> None:
        """Raises BudgetExceeded for the first scope whose total has reached its limit. The totals are read again on
        every check, so a call that another agent made meanwhile counts."""
        now = datetime.now(UTC)
        for scope in self._scopes:
            spent = await scope.spent_usd(self._store, agent_id=self._agent.id, post_id=self._post_id, now=now)
            if scope.used_up(spent):
                raise BudgetExceeded(scope.name, spent, scope.limit_usd)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """A call that passes the check is made in full, so the calls in flight can take a scope slightly over."""
        await self.check()
        return await self._llm.generate(request)
