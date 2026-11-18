import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import HttpUrl

from techdebate.config import Settings
from techdebate.errors import PromptError, ToolError
from techdebate.forum import Forum
from techdebate.llm import FakeLLM, LLMRequest, LLMResponse
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import (
    Evidence,
    LLMCall,
    Message,
    Notification,
    NotificationOutcome,
    ReplyTo,
    Research,
    ResearchPlan,
    SpeakDecision,
    User,
)
from techdebate.orchestrator import AgentLoop, Visit
from techdebate.roles import RoleRegistry, RoleSpec
from techdebate.store import Store
from techdebate.tools import ToolInput, ToolRegistry
from techdebate.tree import Tree
from techdebate.users import sync_users

_RESEARCH = ResearchPlan(speak=True, queries={"stub": ["rust lang"]}, reason="In my area.")
_SKIP = ResearchPlan(speak=False, queries={"stub": []}, reason="Not my area.")
_COMMENT = SpeakDecision(reason="The data is new.", reply_to=[ReplyTo(place=0, point="Downloads doubled.")])
_PASS = SpeakDecision(reason="Nothing new.", reply_to=[])
_PRICES = {"research.v1": 0.01, "decide.v2": 0.02, "developer.v1": 0.25}


class _StubInput(ToolInput):
    query: str


class _StubTool:
    name = "stub"
    description = "Finds things"
    input_model = _StubInput

    def __init__(self, error: ToolError | None = None) -> None:
        self.queries: list[str] = []
        self._error = error

    async def run(self, input: _StubInput) -> list[Evidence]:
        self.queries.append(input.query)
        if self._error:
            raise self._error
        return [
            Evidence(source="stub", query=input.query, url=HttpUrl("https://example.com/x"), title="x", summary="found")
        ]


class _PricedLLM(FakeLLM):
    """Each call costs a set amount, by prompt."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await super().generate(request)
        return response.model_copy(update={"cost_usd": _PRICES[request.prompt_id or ""]})


def _role(name: str, prompt_id: str = "developer.v1") -> RoleSpec:
    return RoleSpec(
        name=name, display_name=name.title(), question="Who uses it?", tool_names=("stub",), prompt_id=prompt_id
    )


def _by(prompt_id: str, name: str) -> Predicate:
    def matches(request: LLMRequest) -> bool:
        return request.prompt_id == prompt_id and request.system.startswith(f"You are the {name.title()},")

    return matches


def _plans(name: str, plan: ResearchPlan) -> tuple[Predicate, Reply]:
    return _by("research.v1", name), plan


def _decides(name: str, decision: SpeakDecision) -> tuple[Predicate, Reply]:
    return _by("decide.v2", name), decision


def _writes(name: str, text: str = "Developers use it.") -> tuple[Predicate, Reply]:
    return _by("developer.v1", name), text


def _comments(name: str, text: str = "Developers use it.") -> list[tuple[Predicate, Reply]]:
    """The calls of an agent that researches and comments on the post."""
    return [_plans(name, _RESEARCH), _decides(name, _COMMENT), _writes(name, text)]


def _calls(llm: FakeLLM, name: str) -> list[str | None]:
    """The prompt IDs of one agent's calls, in order."""
    return [r.prompt_id for r in llm.calls if r.system.startswith(f"You are the {name.title()},")]


@dataclass
class _World:
    store: Store
    me: User
    agents: dict[str, User]
    forum: Forum
    roles: RoleRegistry
    tool: _StubTool

    def loop(self, llm: FakeLLM, **settings: Any) -> AgentLoop:
        return AgentLoop(
            self.store,
            self.forum,
            self.agents,
            llm=llm,
            roles=self.roles,
            tools=ToolRegistry([self.tool]),
            settings=Settings(_env_file=None, **settings),  # pyright: ignore[reportCallIssue]
        )

    async def post(self, text: str = "Rust") -> Message:
        return await self.forum.create_post(self.me, text)

    async def reply(self, parent: Message, text: str) -> Message:
        return await self.forum.create_comment(self.me, parent.id, text)

    async def tree(self, post: Message) -> Tree:
        found = await self.store.get_post(post.id)
        assert found is not None
        return Tree.of(found)

    async def notifications(self, name: str) -> list[Notification]:
        return await self.store.list_notifications(self.agents[name].id, include_closed=True)


async def _world(store: Store, *names: str, tool: _StubTool | None = None) -> _World:
    roles = RoleRegistry([_role(name) for name in names or ("developer",)])
    users = await sync_users(store, roles, "me")
    forum = Forum(store, users.agents.values())
    return _World(store, users.me, users.agents, forum, roles, tool or _StubTool())


def _outcomes(notifications: list[Notification]) -> list[NotificationOutcome | None]:
    return [notification.outcome for notification in notifications]


# A visit


async def test_an_agent_researches_decides_and_comments_on_the_post(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    llm = FakeLLM(rules=_comments("developer", "Developers use [it](https://example.com/x)."))

    [visit] = await world.loop(llm).run()

    [comment] = (await world.tree(post)).comments
    assert (comment.author_id, comment.parent_id, comment.text) == (
        world.agents["developer"].id,
        post.id,
        "Developers use [it](https://example.com/x).",
    )
    assert visit == Visit(
        agent=world.agents["developer"],
        post_id=post.id,
        outcome=NotificationOutcome.COMMENTED,
        reason="The data is new.",
        notifications=1,
        comments=(1,),
    )
    assert _outcomes(await world.notifications("developer")) == [NotificationOutcome.COMMENTED]
    assert world.tool.queries == ["rust lang"]
    research = await store.get_research(world.agents["developer"].id, post.id)
    assert research is not None and research.results[0].query == "rust lang"
    # Every call is stored, for the agent and the post.
    calls = await store.list_llm_calls(post.id)
    assert [(c.agent_id, c.prompt_id) for c in calls] == [
        (world.agents["developer"].id, prompt_id) for prompt_id in ("research.v1", "decide.v2", "developer.v1")
    ]
    research_call, decide_call, write_call = llm.calls
    overview = "Post by me:\nRust\n\n  No comments yet."
    assert research_call.messages[0].content == f"{overview}\n\nNew to you: the post."
    assert decide_call.messages[0].content.startswith(
        f"{overview}\n\nNew to you: the post.\nYou can comment on: 0 (a top-level comment on the post).\n\nTool"
    )
    assert write_call.messages[0].content.endswith(
        "Your comment: a top-level comment on the post.\nYour point: Downloads doubled."
    )


async def test_a_pass_at_the_research_call_searches_and_saves_nothing(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    llm = FakeLLM(rules=[_plans("developer", _SKIP)])

    [visit] = await world.loop(llm).run()

    assert (visit.outcome, visit.reason, visit.at_research) == (NotificationOutcome.PASSED, "Not my area.", True)
    assert world.tool.queries == []
    assert await store.get_research(world.agents["developer"].id, post.id) is None
    assert _outcomes(await world.notifications("developer")) == [NotificationOutcome.PASSED]


async def test_a_pass_at_the_decide_call_keeps_the_research(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    llm = FakeLLM(rules=[_plans("developer", _RESEARCH), _decides("developer", _PASS)])

    [visit] = await world.loop(llm).run()

    assert (visit.outcome, visit.reason, visit.at_research) == (NotificationOutcome.PASSED, "Nothing new.", False)
    assert (await world.tree(post)).comments == ()
    assert await store.get_research(world.agents["developer"].id, post.id) is not None
    assert _outcomes(await world.notifications("developer")) == [NotificationOutcome.PASSED]


async def test_research_is_done_once_and_reused_on_a_later_visit(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    await world.loop(FakeLLM(rules=_comments("developer"))).run()
    [comment] = (await world.tree(post)).comments
    await world.reply(comment, "Are those downloads real?")
    reply = SpeakDecision(reason="A question.", reply_to=[ReplyTo(place=2, point="Mostly CI.")])
    llm = FakeLLM(rules=[_decides("developer", reply), _writes("developer", "Mostly CI.")])

    [visit] = await world.loop(llm).run()

    assert _calls(llm, "developer") == ["decide.v2", "developer.v1"]  # no research call
    assert world.tool.queries == ["rust lang"]  # searched once, on the first visit
    assert "https://example.com/x" in llm.calls[0].messages[0].content  # the saved report
    assert (visit.outcome, visit.comments) == (NotificationOutcome.COMMENTED, (3,))
    assert (await world.tree(post)).comments[2].parent_id == (await world.tree(post)).comments[1].id


async def test_a_direct_question_is_new_to_the_agent_and_shown_in_full(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    await world.loop(FakeLLM(rules=_comments("developer"))).run()
    [comment] = (await world.tree(post)).comments
    question = "@developer " + "is this real? " * 40
    await world.reply(comment, question)
    llm = FakeLLM(rules=[_decides("developer", _PASS)])

    await world.loop(llm).run()

    content = llm.calls[0].messages[0].content
    assert "New to you: #2.\nA user asked you directly in #2." in content
    assert question.strip() in content  # not cut, though it is a long reply


async def _each_commented(world: _World, post: Message) -> None:
    """Every agent has researched the post and has a top-level comment on it, and nothing is pending."""
    for name, agent in world.agents.items():
        await world.store.add_research(Research(agent_id=agent.id, post_id=post.id, results=[]))
        await world.forum.create_comment(agent, post.id, f"{name.title()} view.")
    for agent in world.agents.values():
        pending = await world.store.list_notifications(agent.id)
        await world.store.close_notifications([n.id for n in pending], NotificationOutcome.PASSED)


def _decide_call(llm: FakeLLM, name: str) -> str:
    [content] = [r.messages[0].content for r in llm.calls if _by("decide.v2", name)(r)]
    return content


async def test_the_decide_call_names_the_other_agents_a_user_asked_in_a_comment(store: Store) -> None:
    world = await _world(store, "developer", "scientist", "pollster")
    post = await world.post()
    await _each_commented(world, post)
    await world.reply(post, "@Scientist and @pollster, is there any data on this?")
    llm = FakeLLM(rules=[(lambda r: r.prompt_id == "decide.v2", _PASS)])

    await world.loop(llm).run()

    assert (
        "New to you: #4.\n#4 is a question from a user to the Scientist and the Pollster, not to you.\nYou can"
        in _decide_call(llm, "developer")
    )
    for asked in ("scientist", "pollster"):
        assert "New to you: #4.\nA user asked you directly in #4.\nYou can" in _decide_call(llm, asked)


async def test_a_post_that_names_an_agent_is_still_open_to_the_others(store: Store) -> None:
    world = await _world(store, "developer", "scientist")
    await world.post("@scientist is there research on Rust?")
    llm = FakeLLM(rules=[(lambda r: r.prompt_id == "research.v1", _RESEARCH), (lambda _: True, _PASS)])

    await world.loop(llm).run()

    assert "asked the" not in _decide_call(llm, "developer")
    assert "A user asked you directly in the post." in _decide_call(llm, "scientist")


async def test_an_agent_naming_another_agent_asks_no_one(store: Store) -> None:
    world = await _world(store, "developer", "scientist", "pollster")
    post = await world.post()
    await _each_commented(world, post)
    await world.forum.create_comment(world.agents["scientist"], post.id, "@pollster, what do the videos say?")
    llm = FakeLLM(rules=[(lambda r: r.prompt_id == "decide.v2", _PASS)])

    await world.loop(llm).run()

    assert "New to you: #4.\nYou can" in _decide_call(llm, "developer")
    assert "asked" not in _decide_call(llm, "pollster")


async def test_every_notification_on_a_post_is_handled_in_one_visit_oldest_post_first(store: Store) -> None:
    world = await _world(store)
    first = await world.post("Rust")
    second = await world.post("Go")
    rules = [_plans("developer", _SKIP)]
    await world.loop(FakeLLM(rules=rules)).run()  # both posts looked at and passed
    # An agent that passed is only notified when mentioned: two mentions on the first post, one on the second.
    await world.reply(first, "@developer what about crates?")
    await world.reply(second, "@developer and Go modules?")
    await world.reply(first, "@developer any numbers?")
    llm = FakeLLM(rules=rules)

    visits = await world.loop(llm).run()

    assert [(v.post_id, v.notifications) for v in visits] == [(first.id, 2), (second.id, 1)]
    assert "New to you: #1, #2.\nA user asked you directly in #1 and #2." in llm.calls[0].messages[0].content
    assert all(n.outcome is NotificationOutcome.PASSED for n in await world.notifications("developer"))


def _researching_go(request: LLMRequest) -> bool:
    return request.prompt_id == "research.v1" and "Post by me:\nGo" in request.messages[0].content


async def test_research_is_kept_apart_per_post(store: Store) -> None:
    tool = _StubTool()
    world = await _world(store, tool=tool)
    rust = await world.post("Rust")
    go = await world.post("Go")
    go_research = ResearchPlan(speak=True, queries={"stub": ["go lang"]}, reason="In my area.")
    rules: list[tuple[Predicate, Reply]] = [(_researching_go, go_research), *_comments("developer")]
    await world.loop(FakeLLM(rules=rules)).run()  # one visit per post, each researching it
    assert tool.queries == ["rust lang", "go lang"]

    developer = world.agents["developer"].id
    for post, query in ((rust, "rust lang"), (go, "go lang")):
        research = await store.get_research(developer, post.id)
        assert research is not None
        assert [result.query for result in research.results] == [query]

    # A mention on the first post later goes straight to the decide call with that post's research, not the other's.
    await world.reply(rust, "@developer and the crates?")
    llm = FakeLLM(rules=[_decides("developer", _PASS)])
    await world.loop(llm).run()
    assert _calls(llm, "developer") == ["decide.v2"]
    assert "rust lang" in llm.calls[0].messages[0].content
    assert "go lang" not in llm.calls[0].messages[0].content
    assert tool.queries == ["rust lang", "go lang"]


# Limits


async def test_a_notification_in_a_full_thread_is_closed_without_an_ai_call(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    loop_llm = FakeLLM(rules=_comments("developer"))
    await world.loop(loop_llm, max_comments_per_thread=1).run()
    [comment] = (await world.tree(post)).comments
    await world.reply(comment, "Not true.")  # in thread #1, which the agent has filled
    llm = FakeLLM()

    [visit] = await world.loop(llm, max_comments_per_thread=1).run()

    assert llm.calls == []
    assert (visit.outcome, visit.reason) == (NotificationOutcome.LIMIT, "no place left on the post")
    assert _outcomes(await world.notifications("developer")) == [
        NotificationOutcome.COMMENTED,
        NotificationOutcome.LIMIT,
    ]


async def test_only_the_notifications_in_a_full_thread_are_closed_and_the_others_acted_on(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    await world.loop(FakeLLM(rules=_comments("developer")), max_comments_per_thread=1).run()
    [comment] = (await world.tree(post)).comments
    await world.reply(comment, "Not true.")  # #2, in the full thread #1
    await world.reply(post, "Another view.")  # #3, a new thread
    llm = FakeLLM(rules=[_decides("developer", _PASS)])

    [visit] = await world.loop(llm, max_comments_per_thread=1).run()

    assert (visit.outcome, visit.notifications) == (NotificationOutcome.PASSED, 2)
    content = llm.calls[0].messages[0].content
    assert "New to you: #3.\nYou can comment on: #3." in content
    assert "You have reached your limit of 1 comments in thread #1." in content
    assert _outcomes(await world.notifications("developer"))[1:] == [
        NotificationOutcome.LIMIT,
        NotificationOutcome.PASSED,
    ]


async def test_a_limit_visit_names_the_full_threads(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    await world.loop(FakeLLM(rules=_comments("developer")), max_comments_per_thread=1).run()
    [comment] = (await world.tree(post)).comments
    await world.reply(post, "Another view.")  # #2, open to the agent
    await world.store.close_notifications(
        [n.id for n in await world.notifications("developer")], NotificationOutcome.SEEN
    )
    await world.reply(comment, "Not true.")

    [visit] = await world.loop(FakeLLM(), max_comments_per_thread=1).run()

    assert (visit.outcome, visit.reason) == (NotificationOutcome.LIMIT, "thread #1 is full")


# Several agents and rounds


async def test_two_agents_act_at_the_same_time(store: Store) -> None:
    world = await _world(store, "first", "second")
    post = await world.post()
    started = {"first": asyncio.Event(), "second": asyncio.Event()}

    class _Meeting(FakeLLM):
        """Each agent's research call waits until the other's has started, which only works if both run at once."""

        async def generate(self, request: LLMRequest) -> LLMResponse:
            if request.prompt_id == "research.v1":
                me, other = ("first", "second") if "You are the First" in request.system else ("second", "first")
                started[me].set()
                await asyncio.wait_for(started[other].wait(), timeout=5)
            return await super().generate(request)

    def later_look(request: LLMRequest) -> bool:  # on the other agent's comment, which notifies the one before it
        return request.prompt_id == "decide.v2" and "New to you: the post" not in request.messages[0].content

    llm = _Meeting(
        rules=[(later_look, _PASS), *_comments("first", "First view."), *_comments("second", "Second view.")]
    )

    visits = await world.loop(llm).run()

    commented = [v.agent.username for v in visits if v.outcome is NotificationOutcome.COMMENTED]
    assert sorted(commented) == ["first", "second"]
    assert sorted(c.text for c in (await world.tree(post)).comments) == ["First view.", "Second view."]


async def test_comments_written_after_an_agent_stopped_start_another_round(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = await _world(store, "first", "second")
    post = await world.post()
    first_stopped = asyncio.Event()
    visit = AgentLoop.visit

    async def watched(self: AgentLoop, role: RoleSpec) -> Visit | None:
        result = await visit(self, role)
        if role.name == "first" and result is None:
            first_stopped.set()
        return result

    monkeypatch.setattr(AgentLoop, "visit", watched)

    class _Late(FakeLLM):
        """The second agent starts only once the first has stopped, so its comment comes after."""

        async def generate(self, request: LLMRequest) -> LLMResponse:
            if request.prompt_id == "research.v1" and "You are the Second" in request.system:
                await asyncio.wait_for(first_stopped.wait(), timeout=5)
            return await super().generate(request)

    def second_look(request: LLMRequest) -> bool:
        return _by("decide.v2", "first")(request) and "New to you: #2." in request.messages[0].content

    llm = _Late(rules=[(second_look, _PASS), *_comments("first"), *_comments("second")])

    visits = await world.loop(llm).run()

    # The second agent's top-level comment notified the first, which commented on the post, in a second round.
    assert [(v.agent.username, v.outcome) for v in visits] == [
        ("first", NotificationOutcome.COMMENTED),
        ("second", NotificationOutcome.COMMENTED),
        ("first", NotificationOutcome.PASSED),
    ]
    assert "New to you: #2." in llm.calls[-1].messages[0].content
    for name in ("first", "second"):
        assert all(not n.pending for n in await world.notifications(name))
    assert len((await world.tree(post)).comments) == 2


async def test_run_returns_at_once_when_nothing_is_pending(store: Store) -> None:
    world = await _world(store)
    llm = FakeLLM()

    assert await world.loop(llm).run() == []
    assert llm.calls == []


async def test_each_visit_is_reported_as_it_ends_and_logged(store: Store, caplog: pytest.LogCaptureFixture) -> None:
    world = await _world(store)
    post = await world.post()
    seen: list[Visit] = []

    with caplog.at_level(logging.INFO, logger="techdebate.orchestrator"):
        visits = await world.loop(FakeLLM(rules=[_plans("developer", _SKIP)])).run(on_visit=seen.append)

    assert seen == visits
    assert caplog.records[-1].getMessage() == (
        f"agent visited post_id={post.id} agent=developer outcome=passed notifications=1 comments=[] "
        "reason=Not my area."
    )


# Failures and cost


async def test_a_failed_visit_is_closed_as_failed_and_the_other_agents_go_on(store: Store) -> None:
    world = await _world(store, "first", "second")
    post = await world.post()
    llm = FakeLLM(rules=[_plans("first", _RESEARCH), *_comments("second")])  # no decide reply for the first

    visits = {v.agent.username: v for v in await world.loop(llm).run()}

    assert visits["first"].outcome is NotificationOutcome.FAILED
    assert "no scripted reply" in visits["first"].reason
    assert visits["second"].outcome is NotificationOutcome.COMMENTED
    assert _outcomes(await world.notifications("first")) == [NotificationOutcome.FAILED]
    assert await store.get_research(world.agents["first"].id, post.id) is not None  # saved before the decide call


class _Buggy(FakeLLM):
    """The first agent's decide call hits a bug: an error that is neither LLMError nor BudgetExceeded."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if _by("decide.v2", "first")(request):
            raise KeyError("id")
        return await super().generate(request)


async def test_an_unexpected_error_is_logged_and_closed_as_failed_and_the_other_agents_go_on(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    world = await _world(store, "first", "second")
    post = await world.post()
    llm = _Buggy(rules=[_plans("first", _RESEARCH), *_comments("second")])

    with caplog.at_level(logging.ERROR, logger="techdebate.orchestrator"):
        visits = {v.agent.username: v for v in await world.loop(llm).run()}

    assert (visits["first"].outcome, visits["first"].reason) == (
        NotificationOutcome.FAILED,
        "unexpected error: KeyError('id')",
    )
    assert visits["second"].outcome is NotificationOutcome.COMMENTED
    assert _outcomes(await world.notifications("first")) == [NotificationOutcome.FAILED]
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert record.getMessage() == f"visit failed unexpectedly post_id={post.id} agent=first"
    assert record.exc_info is not None  # the traceback is logged


async def test_an_agent_acts_again_after_an_unexpected_error(store: Store) -> None:
    world = await _world(store, "first")
    post = await world.post()
    await world.loop(_Buggy(rules=[_plans("first", _RESEARCH)])).run()
    await world.reply(post, "@first any numbers?")
    llm = FakeLLM(rules=[_decides("first", _PASS)])

    [visit] = await world.loop(llm).run()

    assert visit.outcome is NotificationOutcome.PASSED
    assert _calls(llm, "first") == ["decide.v2"]  # the research saved before the bug is used


async def test_a_visit_that_fails_after_a_comment_keeps_it(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    await world.reply(post, "My view.")  # #1, by me
    both = SpeakDecision(
        reason="Two points.", reply_to=[ReplyTo(place=0, point="Own."), ReplyTo(place=1, point="Answer.")]
    )
    llm = FakeLLM(
        rules=[
            _plans("developer", _RESEARCH),
            _decides("developer", both),
            (lambda r: _by("developer.v1", "developer")(r) and "Own." in r.messages[0].content, "My own view."),
        ]
    )

    [visit] = await world.loop(llm).run()

    assert (visit.outcome, visit.comments) == (NotificationOutcome.FAILED, (2,))
    assert [c.text for c in (await world.tree(post)).comments] == ["My view.", "My own view."]


async def test_an_agent_still_comments_when_every_tool_fails(store: Store) -> None:
    world = await _world(store, tool=_StubTool(error=ToolError("stub: down")))
    await world.post()
    llm = FakeLLM(rules=_comments("developer", "It is widely used."))

    [visit] = await world.loop(llm).run()

    assert visit.outcome is NotificationOutcome.COMMENTED
    assert 'stub, query "rust lang": the tool failed' in llm.calls[1].messages[0].content


async def test_a_bad_prompt_id_fails_before_any_ai_call(store: Store) -> None:
    world = await _world(store, "first")
    world.roles = RoleRegistry([_role("first"), _role("second", prompt_id="malformed")])
    llm = FakeLLM()

    with pytest.raises(PromptError, match="invalid prompt ID"):
        world.loop(llm)

    assert llm.calls == []


# The post budget


async def test_the_agent_stops_when_the_post_budget_is_used_up(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    llm = _PricedLLM(rules=_comments("developer"))

    # The research call ($0.01) and the decide call ($0.02) pass the check, and the write call finds $0.03 spent.
    [visit] = await world.loop(llm, post_budget_usd=0.02).run()

    assert (visit.outcome, visit.comments) == (NotificationOutcome.BUDGET, ())
    assert visit.reason == "the post has spent $0.0300 of its $0.02 budget"
    assert visit.cost_usd == pytest.approx(0.03)
    assert _calls(llm, "developer") == ["research.v1", "decide.v2"]
    assert _outcomes(await world.notifications("developer")) == [NotificationOutcome.BUDGET]
    assert await store.get_research(world.agents["developer"].id, post.id) is not None
    assert (await world.tree(post)).comments == ()


async def test_later_notifications_on_a_post_out_of_budget_cost_nothing_and_i_can_still_comment(
    store: Store,
) -> None:
    world = await _world(store, "first", "second")
    post = await world.post()
    llm = _PricedLLM(rules=[*_comments("first"), *_comments("second")])
    await world.loop(llm, post_budget_usd=0.02).run()
    calls = len(llm.calls)

    mine = await world.reply(post, "@first @second what about the downloads?")
    visits = await world.loop(llm, post_budget_usd=0.02).run()

    assert len(llm.calls) == calls  # no AI call
    assert {(v.agent.username, v.outcome, v.notifications) for v in visits} == {
        ("first", NotificationOutcome.BUDGET, 1),
        ("second", NotificationOutcome.BUDGET, 1),
    }
    assert all(v.reason.startswith("the post has spent") for v in visits)
    for name in ("first", "second"):
        assert _outcomes(await world.notifications(name)) == [NotificationOutcome.BUDGET] * 2
    assert (await world.tree(post)).comments == (mine,)


async def test_a_visit_stopped_by_the_budget_keeps_the_comments_it_wrote(store: Store) -> None:
    world = await _world(store)
    post = await world.post()
    await world.reply(post, "My view.")  # #1, by me
    both = SpeakDecision(
        reason="Two points.", reply_to=[ReplyTo(place=0, point="Own."), ReplyTo(place=1, point="Answer.")]
    )
    llm = _PricedLLM(rules=[_plans("developer", _RESEARCH), _decides("developer", both), _writes("developer")])

    # $0.03 before the first write, $0.28 after it.
    [visit] = await world.loop(llm, post_budget_usd=0.27).run()

    assert (visit.outcome, visit.comments) == (NotificationOutcome.BUDGET, (2,))
    assert _calls(llm, "developer") == ["research.v1", "decide.v2", "developer.v1"]
    assert [c.text for c in (await world.tree(post)).comments] == ["My view.", "Developers use it."]


# The budgets per agent and per day


async def _spent(world: _World, name: str, post: Message, cost_usd: float, at: datetime | None = None) -> None:
    """An AI call the agent made on the post, stored as if made at `at` (now by default)."""
    call = LLMCall(
        agent_id=world.agents[name].id,
        post_id=post.id,
        prompt_id="developer.v1",
        model="m",
        input_tokens=1,
        output_tokens=1,
        cost_usd=cost_usd,
        duration_s=0.1,
        created_at=at or datetime.now(UTC),
    )
    await world.store.add_llm_call(call)


async def _handled(world: _World) -> None:
    """Closes every agent's pending notifications, as if the agents had passed."""
    for user in world.agents.values():
        pending = await world.store.list_notifications(user.id)
        await world.store.close_notifications([n.id for n in pending], NotificationOutcome.PASSED)


async def test_an_agent_out_of_its_budget_for_today_stops_while_the_others_go_on(store: Store) -> None:
    world = await _world(store, "first", "second")
    earlier = await world.post("Go")
    await _handled(world)
    await _spent(world, "first", earlier, 0.30)  # on another post, which counts too
    post = await world.post()
    llm = _PricedLLM(rules=[*_comments("first"), *_comments("second")])

    visits = await world.loop(llm, agent_budget_usd=0.30).run()  # the second's comment costs $0.28

    assert {(v.agent.username, v.outcome, v.reason) for v in visits} == {
        ("first", NotificationOutcome.BUDGET, "the agent has spent $0.3000 of its $0.30 budget for today"),
        ("second", NotificationOutcome.COMMENTED, "The data is new."),
    }
    assert _calls(llm, "first") == []
    assert _calls(llm, "second") == ["research.v1", "decide.v2", "developer.v1"]
    assert [c.author_id for c in (await world.tree(post)).comments] == [world.agents["second"].id]
    assert _outcomes(await world.notifications("first")) == [NotificationOutcome.PASSED, NotificationOutcome.BUDGET]


async def test_all_agents_stop_on_every_post_at_the_daily_budget(store: Store) -> None:
    world = await _world(store, "first", "second")
    earlier = await world.post("Go")
    await _spent(world, "first", earlier, 0.20)
    await _spent(world, "second", earlier, 0.10)
    await world.post()
    llm = _PricedLLM(rules=[*_comments("first"), *_comments("second")])

    visits = await world.loop(llm, daily_budget_usd=0.30).run()

    assert llm.calls == []
    assert {(v.agent.username, v.outcome) for v in visits} == {
        ("first", NotificationOutcome.BUDGET),
        ("second", NotificationOutcome.BUDGET),
    }
    assert {v.reason for v in visits} == {"the agents have spent $0.3000 of their $0.30 budget for today"}
    for name in ("first", "second"):
        assert _outcomes(await world.notifications(name)) == [NotificationOutcome.BUDGET] * 2


async def test_the_next_day_the_agents_act_on_new_notifications_and_not_on_those_closed_for_the_budget(
    store: Store,
) -> None:
    world = await _world(store)
    yesterday = datetime.now(UTC) - timedelta(days=1)
    earlier = await world.post("Go")
    await _spent(world, "developer", earlier, 0.30, at=yesterday)  # yesterday's budgets were used up
    [closed] = await world.notifications("developer")
    await store.close_notifications([closed.id], NotificationOutcome.BUDGET)  # as yesterday's visit did
    post = await world.post()
    llm = _PricedLLM(rules=_comments("developer"))

    [visit] = await world.loop(llm, agent_budget_usd=0.30, daily_budget_usd=0.30).run()

    assert (visit.post_id, visit.outcome) == (post.id, NotificationOutcome.COMMENTED)
    assert _outcomes(await world.notifications("developer")) == [
        NotificationOutcome.BUDGET,
        NotificationOutcome.COMMENTED,
    ]
    assert (await world.tree(earlier)).comments == ()


async def test_a_visit_costs_its_stored_calls(store: Store) -> None:
    world = await _world(store)
    post = await world.post()

    [visit] = await world.loop(_PricedLLM(rules=_comments("developer"))).run()

    assert visit.cost_usd == pytest.approx(0.01 + 0.02 + 0.25)
    assert sum(c.cost_usd for c in await store.list_llm_calls(post.id)) == pytest.approx(visit.cost_usd)
