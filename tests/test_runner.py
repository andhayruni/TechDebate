import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from techdebate.config import Settings
from techdebate.errors import CommandError
from techdebate.forum import Forum
from techdebate.llm import FakeLLM, LLMRequest, LLMResponse
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import (
    Evidence,
    Message,
    Notification,
    NotificationOutcome,
    ReplyTo,
    ResearchPlan,
    SpeakDecision,
    User,
)
from techdebate.orchestrator import AgentLoop, Visit
from techdebate.roles import RoleRegistry, RoleSpec
from techdebate.runner import AgentRunner, agents_lock
from techdebate.store import Store
from techdebate.tools import ToolInput, ToolRegistry
from techdebate.tree import Tree
from techdebate.users import sync_users

_RESEARCH = ResearchPlan(speak=True, queries={"stub": ["rust lang"]}, reason="In my area.")
_COMMENT = SpeakDecision(reason="The data is new.", reply_to=[ReplyTo(place=0, point="Downloads doubled.")])
_PASS = SpeakDecision(reason="Nothing new.", reply_to=[])
# Long enough that only a wake-up can explain a quick visit, short enough that a wrong test does not hang for long.
_NO_POLL = 60.0


class _StubInput(ToolInput):
    query: str


class _StubTool:
    name = "stub"
    description = "Finds things"
    input_model = _StubInput

    async def run(self, input: _StubInput) -> list[Evidence]:
        return [
            Evidence(source="stub", query=input.query, url=HttpUrl("https://example.com/x"), title="x", summary="found")
        ]


def _role(name: str) -> RoleSpec:
    return RoleSpec(
        name=name, display_name=name.title(), question="Who uses it?", tool_names=("stub",), prompt_id="developer.v1"
    )


def _by(prompt_id: str, name: str) -> Predicate:
    def matches(request: LLMRequest) -> bool:
        return request.prompt_id == prompt_id and request.system.startswith(f"You are the {name.title()},")

    return matches


def _comments(name: str, text: str = "Developers use it.") -> list[tuple[Predicate, Reply]]:
    """An agent that researches and comments on a post the first time, and passes after that."""
    return [
        (_by("research.v1", name), _RESEARCH),
        (lambda r: _by("decide.v2", name)(r) and "New to you: the post." in r.messages[0].content, _COMMENT),
        (_by("decide.v2", name), _PASS),
        (_by("developer.v1", name), text),
    ]


async def _until(condition: Callable[[], Awaitable[bool]], within_s: float = 5.0) -> None:
    """Waits until the condition on the database holds, or fails the test."""
    async with asyncio.timeout(within_s):
        # The condition is read from the database, so there is no event to wait on.
        while not await condition():  # noqa: ASYNC110
            await asyncio.sleep(0.01)


@dataclass
class _World:
    store: Store
    me: User
    agents: dict[str, User]
    forum: Forum
    roles: RoleRegistry

    def runner(self, llm: FakeLLM, *, poll_s: float = _NO_POLL, **settings: Any) -> AgentRunner:
        """A runner that the forum wakes."""
        loop = AgentLoop(
            self.store,
            self.forum,
            self.agents,
            llm=llm,
            roles=self.roles,
            tools=ToolRegistry([_StubTool()]),
            settings=Settings(_env_file=None, **settings),  # pyright: ignore[reportCallIssue]
        )
        runner = AgentRunner(loop, poll_s=poll_s)
        self.forum.subscribe(runner.wake)
        return runner

    async def post(self, text: str = "Rust", forum: Forum | None = None) -> Message:
        return await (forum or self.forum).create_post(self.me, text)

    async def notifications(self, name: str) -> list[Notification]:
        return await self.store.list_notifications(self.agents[name].id, include_closed=True)

    async def handled(self, name: str, count: int = 1) -> bool:
        """Whether the agent has closed `count` notifications and has none pending."""
        notifications = await self.notifications(name)
        return len(notifications) == count and not any(n.pending for n in notifications)

    async def comments(self, post: Message) -> tuple[Message, ...]:
        found = await self.store.get_post(post.id)
        assert found is not None
        return Tree.of(found).comments


async def _world(store: Store, *names: str) -> _World:
    roles = RoleRegistry([_role(name) for name in names or ("developer",)])
    users = await sync_users(store, roles, "me")
    return _World(store, users.me, users.agents, Forum(store, users.agents.values()), roles)


def _outcomes(notifications: list[Notification]) -> list[NotificationOutcome | None]:
    return [notification.outcome for notification in notifications]


# Waiting and waking


async def test_notifications_pending_at_the_start_are_handled_and_the_runner_keeps_running(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    runner = world.runner(FakeLLM(rules=_comments("developer")))

    await runner.start()
    await _until(lambda: world.handled("developer"))
    await asyncio.sleep(0.05)

    assert all(not task.done() for task in runner._tasks)  # pyright: ignore[reportPrivateUsage]
    await runner.stop()
    assert [c.text for c in await world.comments(post)] == ["Developers use it."]
    assert [v.outcome for v in runner.visits] == [NotificationOutcome.COMMENTED]


async def test_a_notification_saved_while_the_runner_waits_wakes_the_agent(store: Store) -> None:
    world = await _world(store)
    runner = world.runner(FakeLLM(rules=_comments("developer")))
    await runner.start()
    await asyncio.sleep(0.05)  # the worker found nothing and waits

    post = await world.post()

    await _until(lambda: world.handled("developer"), within_s=2)  # far sooner than the poll interval
    await runner.stop()
    assert len(await world.comments(post)) == 1


async def test_one_agents_comment_wakes_another(store: Store) -> None:
    world = await _world(store, "first", "second")
    post = await world.post()
    runner = world.runner(FakeLLM(rules=[*_comments("first", "First view."), *_comments("second", "Second view.")]))

    await runner.start()
    # The later top-level comment notifies the agent that commented before it, through a wake-up only.
    await _until(lambda: _settled(world, post), within_s=2)
    await runner.stop()

    assert sorted(c.text for c in await world.comments(post)) == ["First view.", "Second view."]
    later = [n for name in ("first", "second") for n in (await world.notifications(name))[1:]]
    assert later and all(n.outcome is NotificationOutcome.PASSED for n in later)


async def _settled(world: _World, post: Message) -> bool:
    """Both agents have commented, the later comment has notified the earlier agent, and nothing is pending."""
    notifications = [n for name in ("first", "second") for n in await world.notifications(name)]
    return len(await world.comments(post)) == 2 and len(notifications) > 2 and not any(n.pending for n in notifications)


async def test_a_notification_written_by_another_process_is_picked_up_by_polling(store: Store) -> None:
    world = await _world(store)
    runner = world.runner(FakeLLM(rules=_comments("developer")), poll_s=0.05)
    await runner.start()
    await asyncio.sleep(0.05)
    elsewhere = Forum(store, world.agents.values())  # no listener, as in another process

    await world.post(forum=elsewhere)

    await _until(lambda: world.handled("developer"))
    await runner.stop()


async def test_without_polling_a_notification_from_elsewhere_waits(store: Store) -> None:
    world = await _world(store)
    runner = world.runner(FakeLLM(rules=_comments("developer")))
    await runner.start()
    await asyncio.sleep(0.05)

    await world.post(forum=Forum(store, world.agents.values()))
    await asyncio.sleep(0.2)

    assert _outcomes(await world.notifications("developer")) == [None]  # still pending
    await runner.stop()


# Stopping


class _Held(FakeLLM):
    """The write call waits until `release` is set, so a test can act while a visit is in progress."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.writing = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_id == "developer.v1":
            self.writing.set()
            await self.release.wait()
        return await super().generate(request)


async def test_stop_lets_the_visit_in_progress_finish_and_starts_no_other(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    llm = _Held(rules=_comments("developer"))
    runner = world.runner(llm)
    await runner.start()
    await asyncio.wait_for(llm.writing.wait(), timeout=5)
    assert runner.in_progress == 1

    stopping = asyncio.create_task(runner.stop())
    await asyncio.sleep(0.05)
    assert not stopping.done()  # waits for the visit
    await world.post("Go")  # a new notification during the stop starts no visit
    llm.release.set()
    await asyncio.wait_for(stopping, timeout=5)

    assert [c.text for c in await world.comments(post)] == ["Developers use it."]
    assert _outcomes(await world.notifications("developer")) == [NotificationOutcome.COMMENTED, None]
    assert [v.post_id for v in runner.visits] == [post.id]
    assert runner.in_progress == 0


async def test_cancel_ends_the_visit_in_progress_and_leaves_its_notifications_pending(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    llm = _Held(rules=_comments("developer"))
    runner = world.runner(llm)
    await runner.start()
    await asyncio.wait_for(llm.writing.wait(), timeout=5)

    stopping = asyncio.create_task(runner.stop())
    await asyncio.sleep(0.01)
    runner.cancel()
    await asyncio.wait_for(stopping, timeout=5)

    assert await world.comments(post) == ()
    assert _outcomes(await world.notifications("developer")) == [None]
    assert runner.visits == []


async def test_stop_while_waiting_returns_at_once(store: Store) -> None:
    world = await _world(store)
    runner = world.runner(FakeLLM())
    await runner.start()
    await asyncio.sleep(0.05)

    await asyncio.wait_for(runner.stop(), timeout=1)  # not after the poll interval


# Failures


class _BuggyOnce(FakeLLM):
    """The first research call hits a bug."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raised = False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_id == "research.v1" and not self.raised:
            self.raised = True
            raise KeyError("id")
        return await super().generate(request)


async def test_an_agent_keeps_working_after_a_visit_that_raised(store: Store) -> None:
    world = await _world(store)
    first = await world.post("Rust")
    runner = world.runner(_BuggyOnce(rules=_comments("developer")))
    await runner.start()
    await _until(lambda: world.handled("developer"))

    second = await world.post("Go")

    await _until(lambda: world.handled("developer", 2))
    await runner.stop()
    assert [(v.post_id, v.outcome) for v in runner.visits] == [
        (first.id, NotificationOutcome.FAILED),
        (second.id, NotificationOutcome.COMMENTED),
    ]
    assert runner.visits[0].reason == "unexpected error: KeyError('id')"


async def test_an_error_outside_a_visit_is_logged_and_the_worker_tries_again_after_a_poll(
    store: Store, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    world = await _world(store)
    await world.post()
    visit = AgentLoop.visit
    failed: list[bool] = []

    async def flaky(self: AgentLoop, role: RoleSpec) -> Visit | None:
        if not failed:
            failed.append(True)
            raise RuntimeError("database is locked")
        return await visit(self, role)

    monkeypatch.setattr(AgentLoop, "visit", flaky)
    runner = world.runner(FakeLLM(rules=_comments("developer")), poll_s=0.1)

    with caplog.at_level(logging.ERROR, logger="techdebate.runner"):
        await runner.start()
        await _until(lambda: world.handled("developer"))
        await runner.stop()

    assert [v.outcome for v in runner.visits] == [NotificationOutcome.COMMENTED]
    [record] = caplog.records
    assert record.getMessage() == "agent worker failed agent=developer"
    assert record.exc_info is not None


# One runner at a time


def test_a_second_holder_of_the_agents_lock_is_refused_until_the_first_lets_go(db_path: Path) -> None:
    settings = Settings(_env_file=None, db_path=str(db_path))  # pyright: ignore[reportCallIssue]

    with agents_lock(settings):
        with pytest.raises(CommandError, match="the agents are already running on this database"):
            with agents_lock(settings):
                pass

    with agents_lock(settings):  # released
        pass
