"""What the budgets have spent, against their limits: today for all agents and for each agent, and on a post. The
figures come from the stored AI calls, through the budget scopes the visits are checked against, so `techdebate budget`
and the web pages always agree with what stops the agents."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from techdebate.config import Settings
from techdebate.llm import BudgetScope, budget_scopes, end_of_day
from techdebate.models import User
from techdebate.store import Store


@dataclass(frozen=True)
class Spent:
    """What one scope has spent, against its limit."""

    spent_usd: float
    scope: BudgetScope

    @property
    def limit_usd(self) -> float:
        return self.scope.limit_usd

    @property
    def used_up(self) -> bool:
        return self.scope.used_up(self.spent_usd)

    @property
    def left_usd(self) -> float:
        return self.scope.limit_usd - self.spent_usd


@dataclass(frozen=True)
class Today:
    """Today's spending, in UTC days: for all agents together, and for each agent."""

    resets_in: timedelta  # until the daily budgets reset
    all_agents: Spent
    agents: list[tuple[User, Spent]]  # in the order given, which is registry order


def _scopes(settings: Settings) -> dict[str, BudgetScope]:
    return {scope.name: scope for scope in budget_scopes(settings)}


async def spent_today(
    store: Store, agents: Iterable[User], settings: Settings, *, now: datetime | None = None
) -> Today:
    now = now or datetime.now(UTC)
    scopes = _scopes(settings)
    day, per_agent = scopes["day"], scopes["agent"]
    return Today(
        resets_in=end_of_day(now) - now,
        all_agents=Spent(await day.spent_usd(store, now=now), day),
        agents=[
            (agent, Spent(await per_agent.spent_usd(store, agent_id=agent.id, now=now), per_agent)) for agent in agents
        ],
    )


async def spent_on_post(store: Store, post_id: str, settings: Settings) -> Spent:
    scope = _scopes(settings)["post"]
    return Spent(await scope.spent_usd(store, post_id=post_id), scope)


def duration(left: timedelta) -> str:
    """How long until something, in hours and minutes: `7h 12m`."""
    minutes = int(left.total_seconds()) // 60
    return f"{minutes // 60}h {minutes % 60}m"
