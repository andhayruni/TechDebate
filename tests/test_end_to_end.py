"""A whole discussion on one post, through the commands and with no network: the model is scripted, the sources answer
from recorded fixtures, and the database is a temporary file.

The agents act at the same time, so the AI answers are matched by agent and call, not by order. Only the order in which
the first two top-level comments are saved is fixed, since the rest of the discussion depends on it. Everything after
that is ordered by the discussion itself: each reply waits for the comment it answers."""

import asyncio
from collections import Counter
from pathlib import Path
from typing import cast

import pytest
import respx
from respx.models import Call

from techdebate import cli
from techdebate.config import Settings
from techdebate.forum import Forum
from techdebate.llm import FakeLLM, LLMRequest
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import (
    Message,
    Notification,
    NotificationOutcome,
    ReplyTo,
    ResearchPlan,
    SpeakDecision,
    User,
)
from techdebate.store import open_store

_AGENTS = ("Developer", "Scientist", "Pollster")
_WRITE_PROMPTS = {agent: f"{agent.lower()}.v1" for agent in _AGENTS}

# The comments, by number, as the scenario writes them.
_TEXT = {
    1: "Developers build on it: [httpx on PyPI](https://pypi.org/project/httpx/) has heavy downloads.",
    2: "Researchers hardly write about HTTP clients: [one paper](https://arxiv.org/abs/2010.11929) is all I found.",
    3: "Few papers is expected: libraries like this show up in code, not in titles.",
    4: "Agreed, so my counts say nothing about its use, only that it is not a research topic.",
    5: "Then the download numbers are the measure to watch here.",
    6: "@pollster is the public watching videos about it?",
    7: "Some are: [videos](https://www.youtube.com/results?search_query=httpx) get steady views.",
}


def _call(agent: str, prompt_id: str, *contains: str) -> Predicate:
    """A call of one agent: every prompt starts with the agent's display name. With `contains`, the user message must
    hold one of them, which tells apart the calls an agent makes more than once."""

    def matches(request: LLMRequest) -> bool:
        content = request.messages[-1].content
        return (
            request.prompt_id == prompt_id
            and request.system.startswith(f"You are the {agent},")
            and (not contains or any(text in content for text in contains))
        )

    return matches


def _comment(agent: str, place: int, number: int, point: str) -> list[tuple[Predicate, Reply]]:
    """A decide call that picks one place, and the write call there, which becomes comment #number."""
    new = "the post" if place == 0 else f"#{place}"
    where = "a top-level comment on the post" if place == 0 else f"a reply to #{place}"
    return [
        (
            _call(agent, "decide.v1", f"New to you: {new}."),
            SpeakDecision(reason="Something new to add.", reply_to=[ReplyTo(place=place, point=point)]),
        ),
        (_call(agent, _WRITE_PROMPTS[agent], f"Your comment: {where}."), _TEXT[number]),
    ]


_SCRIPT: list[tuple[Predicate, Reply]] = [
    # The first run: the Developer and the Scientist research and comment on the post, and the Pollster passes.
    (
        _call("Developer", "research.v1"),
        ResearchPlan(
            speak=True,
            queries={"github_search": ["httpx"], "github_repo": ["httpx"], "pypi": ["httpx"]},
            reason="Developers use it.",
        ),
    ),
    (
        _call("Scientist", "research.v1"),
        ResearchPlan(speak=True, queries={"arxiv": ["http client"]}, reason="There may be papers."),
    ),
    (
        _call("Pollster", "research.v1", "New to you: the post."),
        ResearchPlan(speak=False, queries={"youtube": []}, reason="Not a public topic."),
    ),
    *_comment("Developer", 0, 1, "Downloads are heavy."),
    *_comment("Scientist", 0, 2, "Few papers."),
    # The two answer each other in thread #2, until the Scientist's thread is full.
    *_comment("Developer", 2, 3, "Libraries show up in code."),
    *_comment("Scientist", 3, 4, "So the counts only show it is not a research topic."),
    *_comment("Developer", 4, 5, "Downloads are the measure."),
    # The second run: my mention brings the Pollster in, and the Developer has nothing new.
    (
        _call("Pollster", "research.v1", "New to you: #6.\nA user asked you directly in #6."),
        ResearchPlan(speak=True, queries={"youtube": ["httpx"]}, reason="A user asked me."),
    ),
    *_comment("Pollster", 6, 7, "Videos get steady views."),
    (
        _call("Developer", "decide.v1", "New to you: #6.", "New to you: #6, #7.", "New to you: #7."),
        SpeakDecision(reason="The question is for the Pollster.", reply_to=[]),
    ),
]


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """A local .env can't change the result. With a YouTube key every agent takes part, and arXiv's spacing between
    requests is off. Two comments per thread keep the exchange short."""
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        YOUTUBE_API_KEY="key",  # pyright: ignore[reportCallIssue]
        arxiv_request_interval_s=0,
        db_path=str(tmp_path / "techdebate.db"),
        max_comments_per_thread=2,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


def _scientist_saves_after_the_developer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first two top-level comments could be saved in either order, and the later one notifies the author of the
    earlier. The Scientist's comments wait until the Developer has saved one, so its comment is #2."""
    create_comment = Forum.create_comment
    developer_saved = asyncio.Event()

    async def in_order(self: Forum, author: User, parent_id: str, text: str) -> Message:
        if author.username == "scientist":
            await asyncio.wait_for(developer_saved.wait(), timeout=5)
        comment = await create_comment(self, author, parent_id, text)
        if author.username == "developer":
            developer_saved.set()
        return comment

    monkeypatch.setattr(Forum, "create_comment", in_order)


def _command(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[str, str]:
    cli.main(list(argv))
    return capsys.readouterr()


def _requests(respx_mock: respx.MockRouter) -> Counter[str]:
    """The requests made so far, by source."""
    sources = {
        "api.github.com": "github",
        "pypi.org": "pypi",
        "pypistats.org": "pypi",
        "export.arxiv.org": "arxiv",
        "www.googleapis.com": "youtube",
    }
    calls = cast("list[Call]", respx_mock.calls)
    return Counter(sources[call.request.url.host] for call in calls)


def _prompts(llm: FakeLLM, agent: str) -> list[str | None]:
    """The prompt IDs of one agent's calls, in order."""
    return [call.prompt_id for call in llm.calls if call.system.startswith(f"You are the {agent},")]


def _tool_report(request: LLMRequest) -> str:
    """The tool report of a write call, without the discussion around it."""
    return request.messages[-1].content.split("\n\nTool results:\n\n")[1].split("\n\nYour comment:")[0]


async def _notifications(settings: Settings) -> dict[str, list[Notification]]:
    """Every notification of every user, closed or not, by username."""
    async with open_store(settings) as store:
        return {
            user.username: await store.list_notifications(user.id, include_closed=True)
            for user in await store.list_users()
        }


async def _researched(settings: Settings) -> dict[str, bool]:
    """Whether each agent has saved research on the post."""
    async with open_store(settings) as store:
        [post] = await store.list_posts()
        agents = [user for user in await store.list_users() if user.username != settings.username]
        return {agent.username: await store.get_research(agent.id, post.id) is not None for agent in agents}


@pytest.mark.usefixtures("mock_github", "mock_pypi", "mock_arxiv", "mock_youtube")
def test_a_whole_discussion(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    respx_mock: respx.MockRouter,
) -> None:
    llm = FakeLLM(rules=_SCRIPT)
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)
    _scientist_saves_after_the_developer(monkeypatch)

    # I post, and the agents look: two research and comment, and one passes. The Scientist's #2 notifies the
    # Developer, and the two answer each other until the Scientist's thread is full.
    out, err = _command(capsys, "post", "httpx")
    assert out == ""
    post = err.removeprefix("post=").removesuffix("\n")  # the short ID that starts every line about the post
    out, err = _command(capsys, "run")

    assert sorted(out.splitlines()) == [
        f"{post} Developer [AI] commented: #1",
        f"{post} Developer [AI] commented: #3",
        f"{post} Developer [AI] commented: #5",
        f"{post} Pollster [AI] passed at research: Not a public topic.",
        f"{post} Scientist [AI] at its limit: thread #2 is full",
        f"{post} Scientist [AI] commented: #2",
        f"{post} Scientist [AI] commented: #4",
    ]
    assert err == "visits=7 comments=5 cost=$0.000000\n"  # nothing else, so no tool failed and no role was left out
    first_run = _requests(respx_mock)
    assert first_run["github"] > 0 and first_run["pypi"] > 0 and first_run["arxiv"] > 0
    assert first_run["youtube"] == 0  # the Pollster passed before searching

    # I reply in the Developer's thread and mention the Pollster, which takes its first look again and answers. The
    # Developer is notified of both, and passes. Its notifications can come in one visit or two, so the lines are a set.
    assert _command(capsys, "reply", "1", _TEXT[6]) == ("", f"post={post} comment=#6\n")
    out, err = _command(capsys, "run")

    assert set(out.splitlines()) == {
        f"{post} Pollster [AI] commented: #7",
        f"{post} Developer [AI] passed: The question is for the Pollster.",
    }
    assert err.endswith(" comments=1 cost=$0.000000\n")
    second_run = _requests(respx_mock)
    assert {source: second_run[source] for source in ("github", "pypi", "arxiv")} == {
        source: first_run[source] for source in ("github", "pypi", "arxiv")
    }
    assert second_run["youtube"] > 0

    # The discussion as I read it, and my notifications: the top-level comments on my post, and the answer in the
    # thread I replied in.
    out, err = _command(capsys, "show")
    assert out == (
        "me:\n"
        "httpx\n"
        "\n"
        "  #1 Developer [AI]:\n"
        f"  {_TEXT[1]}\n"
        "\n"
        "    #6 me:\n"
        f"    {_TEXT[6]}\n"
        "\n"
        "    #7 Pollster [AI], replying to #6:\n"
        f"    {_TEXT[7]}\n"
        "\n"
        "  #2 Scientist [AI]:\n"
        f"  {_TEXT[2]}\n"
        "\n"
        "    #3 Developer [AI]:\n"
        f"    {_TEXT[3]}\n"
        "\n"
        "    #4 Scientist [AI], replying to #3:\n"
        f"    {_TEXT[4]}\n"
        "\n"
        "    #5 Developer [AI], replying to #4:\n"
        f"    {_TEXT[5]}\n"
    )
    assert err == f"post={post} comments=7\n"
    out, err = _command(capsys, "notifications")
    assert out == (
        f"{post} #1 Developer [AI] commented on your post:\n{_TEXT[1]}\n\n"
        f"{post} #2 Scientist [AI] commented on your post:\n{_TEXT[2]}\n\n"
        f"{post} #7 Pollster [AI] replied to #6, in thread #1:\n{_TEXT[7]}\n"
    )
    assert err == "notifications=3\n"

    # Nothing is left pending, and each notification is closed with what its visit did. The Scientist's full thread
    # cost no AI call.
    notifications = asyncio.run(_notifications(settings))
    assert all(not n.pending for ns in notifications.values() for n in ns)
    assert {name: [n.outcome for n in ns] for name, ns in notifications.items()} == {
        "me": [NotificationOutcome.SEEN] * 3,
        "developer": [NotificationOutcome.COMMENTED] * 3 + [NotificationOutcome.PASSED] * 2,  # the post, #2, #4; #6, #7
        "scientist": [NotificationOutcome.COMMENTED] * 2 + [NotificationOutcome.LIMIT],  # the post, #3; #5
        "pollster": [NotificationOutcome.PASSED, NotificationOutcome.COMMENTED],  # the post; #6
    }

    # Each agent researched the post once: its later visits went straight to the decide call with its saved research,
    # and only the Pollster, which had passed at the research call, took its first look again.
    developer = _prompts(llm, "Developer")
    assert developer[:7] == ["research.v1"] + ["decide.v1", "developer.v1"] * 3
    assert developer[7:] in (["decide.v1"], ["decide.v1"] * 2)
    assert _prompts(llm, "Scientist") == ["research.v1"] + ["decide.v1", "scientist.v1"] * 2
    assert _prompts(llm, "Pollster") == ["research.v1", "research.v1", "decide.v1", "pollster.v1"]
    assert asyncio.run(_researched(settings)) == {"developer": True, "scientist": True, "pollster": True}

    # Each agent wrote with its own tools' results only.
    hosts = {"Developer": "github.com", "Scientist": "arxiv.org", "Pollster": "youtube.com"}
    for agent in _AGENTS:
        for write in (call for call in llm.calls if _call(agent, _WRITE_PROMPTS[agent])(call)):
            report = _tool_report(write)
            assert all((host in report) == (owner == agent) for owner, host in hosts.items()), (agent, report)
