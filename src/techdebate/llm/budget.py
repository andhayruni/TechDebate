"""A client that refuses a call once a budget is used up, for one agent and one post. It wraps the recording, so a
refused call reaches neither the provider nor the stored calls."""

from collections.abc import Sequence
from dataclasses import dataclass

from techdebate.config import Settings
from techdebate.errors import BudgetExceeded
from techdebate.llm.base import LLMClient, LLMRequest, LLMResponse
from techdebate.models import User
from techdebate.store.base import Store


@dataclass(frozen=True)
class BudgetScope:
    """What one budget covers. The post scope counts every agent's calls on the post."""

    name: str  # how BudgetExceeded names it, e.g. "post"
    limit_usd: float

    async def spent_usd(self, store: Store, agent: User, post_id: str) -> float:
        return await store.spent_usd(post_id=post_id)


def post_scopes(settings: Settings) -> list[BudgetScope]:
    """The budgets an agent's calls on a post are checked against."""
    return [BudgetScope(name="post", limit_usd=settings.post_budget_usd)]


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
        for scope in self._scopes:
            spent = await scope.spent_usd(self._store, self._agent, self._post_id)
            if spent >= scope.limit_usd:
                raise BudgetExceeded(scope.name, spent, scope.limit_usd)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """A call that passes the check is made in full, so the calls in flight can take a scope slightly over."""
        await self.check()
        return await self._llm.generate(request)
