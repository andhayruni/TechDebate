"""The agent loop. Nobody decides who speaks next: each agent acts on its own notifications, in visits to one post at a
time, and the agents act at the same time, so the order of the comments comes out of that.

A visit takes the agent's oldest pending notification, with its other pending notifications on the same post. It closes
them all when the post, the agent or the day is out of budget, closes the ones the agent may not act on, researches
the post if the agent has not yet, decides where to comment, writes, and closes the notifications with what happened."""

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from techdebate.analyst import Analyst, News
from techdebate.config import Settings
from techdebate.errors import BudgetExceeded, LLMError
from techdebate.forum import Forum
from techdebate.llm import Budget, LLMClient, RecordingLLM, budget_scopes
from techdebate.models import (
    AgentVisit,
    Message,
    Notification,
    NotificationOutcome,
    NotificationReason,
    Research,
    ToolResult,
    User,
    UserKind,
)
from techdebate.notify import find_mentions
from techdebate.places import POST, Places, one_per_thread, place_message, places_for
from techdebate.roles import RoleRegistry, RoleSpec
from techdebate.store import Store
from techdebate.tools import ToolRegistry, recording_tools
from techdebate.tree import Tree
from techdebate.views import overview_for, thread_for

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Visit:
    """What one visit of an agent to a post did, as `run` and `agents` print it."""

    id: str  # of the stored AgentVisit
    agent: User
    post_id: str
    outcome: NotificationOutcome
    reason: str  # the reason of the call that ended it, what was full, or the error
    notifications: int  # how many it closed
    comments: tuple[int, ...] = ()  # the numbers of the comments it wrote
    at_research: bool = False  # a pass at the research call, not at the decide call
    cost_usd: float = 0.0  # its AI calls, failed ones that cost something included


@dataclass
class _Acted:
    """How far a visit got. Filled in as it goes, so a visit that fails keeps the comments it wrote."""

    outcome: NotificationOutcome = NotificationOutcome.PASSED
    reason: str = ""
    at_research: bool = False
    written: list[Message] = field(default_factory=list[Message])


class AgentLoop:
    """Runs the agents in the role registry on the discussion in the store. `agents` maps each role's name to its
    user, from the sync, and `forum` notifies them."""

    def __init__(
        self,
        store: Store,
        forum: Forum,
        agents: Mapping[str, User],
        *,
        llm: LLMClient,
        roles: RoleRegistry,
        tools: ToolRegistry,
        settings: Settings,
    ) -> None:
        self._store = store
        self._forum = forum
        self._agents = agents
        self._llm = llm
        self._roles = roles
        self._tools = tools
        self._settings = settings
        # Built once here, so a bad prompt ID fails before any spending. Each visit builds its own on its recording.
        for role in roles.roles():
            Analyst(role, llm, tools, settings)

    @property
    def roles(self) -> list[RoleSpec]:
        """The agents' roles, in registry order."""
        return self._roles.roles()

    @property
    def agents(self) -> Mapping[str, User]:
        """Each role's agent user, by role name."""
        return self._agents

    async def run(self, on_visit: Callable[[Visit], None] | None = None) -> list[Visit]:
        """Runs the agents in rounds until no agent has a pending notification, and returns every visit in the order
        they ended. A round starts a worker for each agent with pending notifications, and each worker makes visits
        until it has none left. The comments of one round can notify agents that have already stopped, so another
        round follows until nothing is pending. `on_visit` is called as each visit ends."""
        visits: list[Visit] = []

        async def work(role: RoleSpec) -> None:
            while (visit := await self.visit(role)) is not None:
                visits.append(visit)
                if on_visit is not None:
                    on_visit(visit)

        while waiting := [role for role in self._roles.roles() if await self._pending(role)]:
            # A visit closes its notifications whatever goes wrong in it, so only an error outside one, such as the
            # database failing, reaches here: it cancels the other workers and goes up.
            async with asyncio.TaskGroup() as group:
                for role in waiting:
                    group.create_task(work(role))
        return visits

    async def visit(self, role: RoleSpec) -> Visit | None:
        """The agent's next visit: to the post of its oldest pending notification, handling every pending notification
        it has there. None when nothing is pending.

        The visit is stored from its start, and what it does carries its ID: the notifications it closes, its AI calls
        and its searches. It is finished in the transaction that closes its last notifications, so a visit cut off
        (a second Ctrl-C) stays unfinished, with its notifications still pending."""
        agent = self._agents[role.name]
        pending = await self._pending(role)
        if not pending:
            return None
        post_id = pending[0].post_id
        notifications = [notification for notification in pending if notification.post_id == post_id]
        found = await self._store.get_post(post_id)
        assert found is not None, "a notification's post is saved before it"
        tree = Tree.of(found)
        places = places_for(tree, agent.id, max_per_thread=self._settings.max_comments_per_thread)
        stored = AgentVisit(agent_id=agent.id, post_id=post_id)
        await self._store.add_visit(stored)
        recording = RecordingLLM(self._llm, self._store, agent=agent, post_id=post_id, visit_id=stored.id)
        llm = Budget(recording, self._store, budget_scopes(self._settings), agent=agent, post_id=post_id)
        tools = recording_tools(self._tools, self._store, agent=agent, post_id=post_id, visit_id=stored.id)

        # Step 1: no AI call for what the agent may not act on. A budget used up closes the whole post, so it comes
        # before the limits.
        try:
            await llm.check()
        except BudgetExceeded as e:
            acted = _Acted(outcome=NotificationOutcome.BUDGET, reason=str(e))
            return await self._finish(stored, agent, notifications, notifications, acted, cost_usd=0.0)
        closed = [n for n in notifications if _in_full_thread(tree, places, n)]
        active = [n for n in notifications if n not in closed]
        if not active:
            acted = _Acted(outcome=NotificationOutcome.LIMIT, reason=_limit_reason(places))
            return await self._finish(stored, agent, closed, closed, acted, cost_usd=0.0)
        await self._store.close_notifications([n.id for n in closed], NotificationOutcome.LIMIT, visit_id=stored.id)

        acted = _Acted()
        try:
            await self._act(Analyst(role, llm, tools, self._settings), agent, tree, places, active, acted)
        except LLMError as e:
            log.warning("visit failed post_id=%s agent=%s error=%s", post_id, role.name, e)
            acted.outcome, acted.reason = NotificationOutcome.FAILED, str(e)
        except BudgetExceeded as e:
            acted.outcome, acted.reason = NotificationOutcome.BUDGET, str(e)
        except Exception as e:  # a bug: logged with its traceback, without ending the worker or the other agents
            log.exception("visit failed unexpectedly post_id=%s agent=%s", post_id, role.name)
            acted.outcome, acted.reason = NotificationOutcome.FAILED, f"unexpected error: {e!r}"
        return await self._finish(stored, agent, notifications, active, acted, cost_usd=recording.total_usd)

    async def _finish(
        self,
        stored: AgentVisit,
        agent: User,
        notifications: Sequence[Notification],
        closing: Sequence[Notification],
        acted: _Acted,
        *,
        cost_usd: float,
    ) -> Visit:
        """Closes the visit's last notifications (`closing`) with its outcome, and finishes the stored visit, in one
        transaction. `notifications` are all the visit handled."""
        finished = stored.model_copy(
            update={
                "ended_at": datetime.now(UTC),
                "outcome": acted.outcome,
                "reason": acted.reason,
                "at_research": acted.at_research,
                "comment_ids": [comment.id for comment in acted.written],
            }
        )
        async with self._store.transaction() as tx:
            await tx.close_notifications([n.id for n in closing], acted.outcome, visit_id=stored.id)
            await tx.finish_visit(finished)

        comments: tuple[int, ...] = ()
        if acted.written:  # numbered on the post as it is now, since other agents may have written meanwhile
            now = await self._store.get_post(stored.post_id)
            assert now is not None
            comments = tuple(Tree.of(now).number(comment) for comment in acted.written)
        return _logged(
            Visit(
                id=stored.id,
                agent=agent,
                post_id=stored.post_id,
                outcome=acted.outcome,
                reason=acted.reason,
                notifications=len(notifications),
                comments=comments,
                at_research=acted.at_research,
                cost_usd=cost_usd,
            )
        )

    async def _act(
        self,
        analyst: Analyst,
        agent: User,
        tree: Tree,
        places: Places,
        notifications: Sequence[Notification],
        acted: _Acted,
    ) -> None:
        """Steps 2 to 4: research once, decide, and write. LLMError and BudgetExceeded go up, with `acted` as far as
        it got."""
        post_id = tree.post.id
        messages = {message.id: message for message in (tree.post, *tree.comments)}
        about = [messages[notification.message_id] for notification in notifications]
        users = {user.id: user for user in await self._store.list_users()}
        news = News(
            new=tuple(sorted({_place(tree, message) for message in about})),
            asked_in=tuple(
                sorted(
                    {
                        _place(tree, messages[n.message_id])
                        for n in notifications
                        if n.reason is NotificationReason.MENTION
                    }
                )
            ),
            asked_others=self._asked_others(tree, agent, about, users),
        )
        overview = overview_for(tree, users, reply_chars=self._settings.overview_reply_chars, in_full=about)

        results: list[ToolResult]
        research = await self._store.get_research(agent.id, post_id)
        if research is None:
            plan = await analyst.plan(overview, news)
            log.info(
                "agent planned post_id=%s agent=%s speak=%s reason=%s", post_id, agent.username, plan.speak, plan.reason
            )
            if not plan.speak:
                acted.reason, acted.at_research = plan.reason, True
                return
            results = await analyst.gather(plan.queries)
            # Saved before the decide call, so a call that fails does not lose it.
            await self._store.add_research(Research(agent_id=agent.id, post_id=post_id, results=results))
        else:
            results = research.results

        decision = await analyst.decide(overview, results, places, news)
        reply_to = one_per_thread(tree, decision.reply_to)
        log.info(
            "agent decided post_id=%s agent=%s reply_to=%s reason=%s",
            post_id,
            agent.username,
            [item.place for item in reply_to],
            decision.reason,
        )
        acted.reason = decision.reason
        # One after another, all from the discussion the decision saw.
        for item in reply_to:
            parent = place_message(tree, item.place)
            view = overview if item.place == POST else thread_for(tree, users, parent)
            text = await analyst.write(view, item.place, results, item.point)
            acted.written.append(await self._forum.create_comment(agent, parent.id, text))
            acted.outcome = NotificationOutcome.COMMENTED

    def _asked_others(
        self, tree: Tree, agent: User, about: Sequence[Message], users: Mapping[str, User]
    ) -> dict[int, tuple[str, ...]]:
        """The new comments in which a real user mentions other agents in the registry, with their display names in
        the order they are mentioned. A post that mentions an agent is still open to every agent, so it is left out."""
        others = {role.name: role.display_name for role in self._roles.roles() if role.name != agent.username}
        asked: dict[int, tuple[str, ...]] = {}
        for message in about:
            if message.id == tree.post.id or users[message.author_id].kind is not UserKind.REAL:
                continue
            names = tuple(others[name] for name in find_mentions(message.text) if name in others)
            if names:
                asked[tree.number(message)] = names
        return asked

    async def has_pending(self, role: RoleSpec) -> bool:
        """Whether the agent has a notification to act on."""
        return bool(await self._pending(role))

    async def _pending(self, role: RoleSpec) -> list[Notification]:
        return await self._store.list_notifications(self._agents[role.name].id)


def _logged(visit: Visit) -> Visit:
    log.info(
        "agent visited post_id=%s agent=%s outcome=%s notifications=%d comments=%s reason=%s",
        visit.post_id,
        visit.agent.username,
        visit.outcome,
        visit.notifications,
        list(visit.comments),
        visit.reason,
    )
    return visit


def _place(tree: Tree, message: Message) -> int:
    """The place of a message: 0 for the post, n for comment #n."""
    return POST if message.id == tree.post.id else tree.number(message)


def _in_full_thread(tree: Tree, places: Places, notification: Notification) -> bool:
    """Whether the agent may not act on the notification: no place on the post is open to it, or the notification's
    message is in a thread it has filled."""
    if not places.open:
        return True
    if notification.message_id == tree.post.id:
        return False
    comment = next(c for c in tree.comments if c.id == notification.message_id)
    return tree.number(tree.top_level(comment)) in places.full_threads


def _limit_reason(places: Places) -> str:
    if not places.open:
        return "no place left on the post"
    numbers = [f"#{n}" for n in places.full_threads]
    if len(numbers) == 1:
        return f"thread {numbers[0]} is full"
    return f"threads {', '.join(numbers[:-1])} and {numbers[-1]} are full"
