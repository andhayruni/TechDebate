import asyncio

import pytest
from pydantic import HttpUrl

from techdebate.errors import NotFoundError, ToolError
from techdebate.models import AgentVisit, Evidence, ToolCallOutcome, User, UserKind
from techdebate.models import Message as Post
from techdebate.store import Store
from techdebate.tools import RecordingTool, ToolInput, ToolRegistry, recording_tools


class _Input(ToolInput):
    query: str


class _Tool:
    """Answers with what it is given: evidence, or an error to raise."""

    name = "stub"
    description = "Finds things"
    input_model = _Input

    def __init__(self, answer: list[Evidence] | BaseException) -> None:
        self._answer = answer

    async def run(self, input: _Input) -> list[Evidence]:
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._answer


def _evidence(n: int) -> list[Evidence]:
    return [
        Evidence(source="stub", query="rust", url=HttpUrl(f"https://example.com/{i}"), title="x", summary="found")
        for i in range(n)
    ]


async def _visit(store: Store) -> tuple[User, AgentVisit]:
    me = User(username="me", kind=UserKind.REAL)
    developer = User(username="developer", kind=UserKind.AGENT)
    post = Post(author_id=me.id, text="Is Rust growing?")
    await store.add_user(me)
    await store.add_user(developer)
    await store.add_message(post)
    visit = AgentVisit(agent_id=developer.id, post_id=post.id)
    await store.add_visit(visit)
    return developer, visit


def _recording(tool: _Tool, store: Store, agent: User, visit: AgentVisit) -> RecordingTool:
    return RecordingTool(tool, store, agent=agent, post_id=visit.post_id, visit_id=visit.id)


async def test_a_search_with_evidence_is_stored_and_its_evidence_returned(store: Store) -> None:
    developer, visit = await _visit(store)
    evidence = _evidence(3)
    recording = _recording(_Tool(evidence), store, developer, visit)

    assert await recording.run(_Input(query="rust")) == evidence

    [call] = await store.list_tool_calls(visit.post_id)
    assert (call.visit_id, call.agent_id, call.post_id) == (visit.id, developer.id, visit.post_id)
    assert (call.tool, call.query, call.outcome, call.evidence, call.error) == (
        "stub",
        "rust",
        ToolCallOutcome.EVIDENCE,
        3,
        None,
    )
    assert call.duration_s >= 0


async def test_it_looks_like_the_tool_it_wraps(store: Store) -> None:
    developer, visit = await _visit(store)
    recording = _recording(_Tool([]), store, developer, visit)

    assert (recording.name, recording.description, recording.input_model) == ("stub", "Finds things", _Input)


async def test_a_search_that_returns_nothing_is_no_information(store: Store) -> None:
    developer, visit = await _visit(store)

    assert await _recording(_Tool([]), store, developer, visit).run(_Input(query="rust")) == []

    [call] = await store.list_tool_calls(visit.post_id)
    assert (call.outcome, call.evidence, call.error) == (ToolCallOutcome.NO_INFORMATION, 0, None)


@pytest.mark.parametrize(
    ("error", "outcome", "stored"),
    [
        (
            NotFoundError("no package named 'rsut' on PyPI"),
            ToolCallOutcome.NO_INFORMATION,
            "no package named 'rsut' on PyPI",
        ),
        (ToolError("GitHub search rate limit reached"), ToolCallOutcome.FAILED, "GitHub search rate limit reached"),
        (KeyError("id"), ToolCallOutcome.FAILED, "KeyError('id')"),
    ],
)
async def test_a_search_that_raises_is_stored_with_its_error_and_raises_it_unchanged(
    store: Store, error: Exception, outcome: ToolCallOutcome, stored: str
) -> None:
    developer, visit = await _visit(store)

    with pytest.raises(type(error)) as raised:
        await _recording(_Tool(error), store, developer, visit).run(_Input(query="rsut"))

    assert raised.value is error
    [call] = await store.list_tool_calls(visit.post_id)
    assert (call.query, call.outcome, call.evidence, call.error) == ("rsut", outcome, 0, stored)


async def test_a_cancelled_search_stores_nothing(store: Store) -> None:
    developer, visit = await _visit(store)

    with pytest.raises(asyncio.CancelledError):
        await _recording(_Tool(asyncio.CancelledError()), store, developer, visit).run(_Input(query="rust"))

    assert await store.list_tool_calls(visit.post_id) == []


async def test_recording_tools_wraps_every_tool_for_the_visit(store: Store) -> None:
    developer, visit = await _visit(store)

    class _Other(_Tool):
        name = "other"

    tools = recording_tools(
        ToolRegistry([_Tool(_evidence(1)), _Other([])]),
        store,
        agent=developer,
        post_id=visit.post_id,
        visit_id=visit.id,
    )
    for tool in tools.tools():
        await tool.run(_Input(query="rust"))

    assert [tool.name for tool in tools.tools()] == ["stub", "other"]
    assert all(isinstance(tool, RecordingTool) for tool in tools.tools())
    calls = await store.list_tool_calls(visit.post_id)
    assert [(call.tool, call.outcome, call.visit_id) for call in calls] == [
        ("stub", ToolCallOutcome.EVIDENCE, visit.id),
        ("other", ToolCallOutcome.NO_INFORMATION, visit.id),
    ]
