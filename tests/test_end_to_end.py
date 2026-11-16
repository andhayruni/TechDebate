"""`techdebate post` and then `techdebate run` with no network: only the model and the HTTP responses are replaced,
and the database is a temporary file."""

from collections.abc import Callable
from pathlib import Path

import pytest
import respx

from techdebate import cli
from techdebate.config import Settings
from techdebate.llm import FakeLLM, LLMRequest
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import ReplyTo, ResearchPlan, SpeakDecision

_RESEARCH = ResearchPlan(
    speak=True,
    queries={"github_search": ["httpx"], "github_repo": ["httpx"], "pypi": ["httpx"]},
    reason="Developers use this.",
)
_NO_PYPI = ResearchPlan(
    speak=True,
    queries={"github_search": ["httpx"], "github_repo": ["httpx"], "pypi": []},
    reason="Developers use this, but it is not a package.",
)
_SKIP = ResearchPlan(speak=False, queries={"github_search": [], "github_repo": [], "pypi": []}, reason="Not my area.")
_COMMENT_ON_POST = SpeakDecision(
    reason="The download numbers are new.", reply_to=[ReplyTo(place=0, point="Downloads are heavy.")]
)
_NOTHING_NEW = SpeakDecision(reason="The results only repeat the post.", reply_to=[])
_OVERVIEW = "Post by me:\nhttpx\n\n  No comments yet."
_DECIDE_BLOCK = "New to you: the post.\nYou can comment on: 0 (a top-level comment on the post)."
_COMMENT = "Developers use it, with [heavy downloads](https://pypistats.org/packages/httpx)."


def _is(agent: str, prompt_id: str) -> Callable[[LLMRequest], bool]:
    """A call of one agent: every prompt starts with the agent's display name."""

    def matches(request: LLMRequest) -> bool:
        return request.prompt_id == prompt_id and request.system.startswith(f"You are the {agent},")

    return matches


def _is_research(request: LLMRequest) -> bool:
    return _is("Developer", "research.v1")(request)


def _is_decide(request: LLMRequest) -> bool:
    return _is("Developer", "decide.v1")(request)


def _is_write(request: LLMRequest) -> bool:
    return _is("Developer", "developer.v1")(request)


# The Scientist and the Pollster look at every post too. Unless a test says otherwise, they pass at the research call,
# so they make no request.
_OTHERS_PASS: list[tuple[Predicate, Reply]] = [
    (_is("Scientist", "research.v1"), ResearchPlan(speak=False, queries={"arxiv": []}, reason="No research angle.")),
    (_is("Pollster", "research.v1"), ResearchPlan(speak=False, queries={"youtube": []}, reason="Not a public topic.")),
]


@pytest.fixture(autouse=True)
def _no_local_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    """A local .env can't change the result. With a YouTube key every agent takes part, and arXiv's spacing between
    requests is off. The database is a temporary file."""
    db_path = str(tmp_path / "techdebate.db")
    settings = Settings(_env_file=None, YOUTUBE_API_KEY="key", arxiv_request_interval_s=0, db_path=db_path)  # pyright: ignore[reportCallIssue]
    monkeypatch.setattr(cli, "get_settings", lambda: settings)


def _post_and_run(capsys: pytest.CaptureFixture[str], text: str = "httpx") -> tuple[list[str], str]:
    """Creates the post and runs the agents: the visit lines, sorted since the agents act at the same time, and
    stderr."""
    cli.main(["post", text])
    capsys.readouterr()
    cli.main(["run"])
    out, err = capsys.readouterr()
    return sorted(out.splitlines()), err


def _show(capsys: pytest.CaptureFixture[str]) -> str:
    cli.main(["show"])
    return capsys.readouterr().out


def _calls_of(llm: FakeLLM, agent: str) -> list[LLMRequest]:
    return [call for call in llm.calls if call.system.startswith(f"You are the {agent},")]


@pytest.mark.usefixtures("mock_github", "mock_pypi")
def test_the_developer_comments(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    llm = FakeLLM(
        rules=[*_OTHERS_PASS, (_is_research, _RESEARCH), (_is_decide, _COMMENT_ON_POST), (_is_write, _COMMENT)]
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    lines, err = _post_and_run(capsys)

    research, decide, write = _calls_of(llm, "Developer")
    assert (research.prompt_id, decide.prompt_id, write.prompt_id) == ("research.v1", "decide.v1", "developer.v1")
    assert len(llm.calls) == 5  # and the research calls of the Scientist and the Pollster
    assert research.messages[0].content == f"{_OVERVIEW}\n\nNew to you: the post."
    # The write call gets the overview and the tool report the decide call saw, and where it writes what.
    report = decide.messages[0].content.removeprefix(f"{_OVERVIEW}\n\n{_DECIDE_BLOCK}\n\n")
    assert write.messages[0].content == (
        f"{_OVERVIEW}\n\n{report}\n\nYour comment: a top-level comment on the post.\nYour point: Downloads are heavy."
    )
    assert report.startswith('Tool results:\n\ngithub_search, query "httpx":\n')
    for url in (
        "https://github.com/search?q=httpx&type=repositories",  # github_search
        "https://github.com/fastapi/fastapi/commits",  # github_repo
        "https://pypi.org/project/httpx/",  # pypi
    ):
        assert url in report
    for call in (research, decide, write):
        assert "httpx" not in call.system
        assert "You are the Developer" in call.system

    assert lines == [
        "Developer [AI] commented: #1",
        "Pollster [AI] passed at research: Not a public topic.",
        "Scientist [AI] passed at research: No research angle.",
    ]
    assert err == "visits=3 comments=1 cost=$0.000000\n"  # nothing else, so no tool failed and no role was left out
    assert _show(capsys) == f"me:\nhttpx\n\n  #1 Developer [AI]:\n  {_COMMENT}\n"


def test_every_agent_passes_at_the_research_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    llm = FakeLLM(rules=[*_OTHERS_PASS, (_is_research, _SKIP)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    lines, err = _post_and_run(capsys)

    assert [request.prompt_id for request in llm.calls] == ["research.v1"] * 3
    assert respx_mock.calls.call_count == 0
    assert "Developer [AI] passed at research: Not my area." in lines
    assert err == "visits=3 comments=0 cost=$0.000000\n"
    assert _show(capsys) == "me:\nhttpx\n\n  No comments.\n"


@pytest.mark.usefixtures("mock_github", "mock_pypi")
def test_the_developer_researches_and_then_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    llm = FakeLLM(rules=[*_OTHERS_PASS, (_is_research, _RESEARCH), (_is_decide, _NOTHING_NEW)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    lines, err = _post_and_run(capsys)

    assert [request.prompt_id for request in _calls_of(llm, "Developer")] == ["research.v1", "decide.v1"]
    assert respx_mock.calls.call_count > 0  # the searches ran
    assert "Developer [AI] passed: The results only repeat the post." in lines
    assert err == "visits=3 comments=0 cost=$0.000000\n"
    assert _show(capsys) == "me:\nhttpx\n\n  No comments.\n"


@pytest.mark.usefixtures("mock_github")
def test_the_developer_leaves_out_a_tool_that_does_not_fit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = FakeLLM(
        rules=[*_OTHERS_PASS, (_is_research, _NO_PYPI), (_is_decide, _COMMENT_ON_POST), (_is_write, _COMMENT)]
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    _, err = _post_and_run(capsys)  # a request to PyPI would have no mock, and fail the test

    assert _calls_of(llm, "Developer")[1].messages[0].content.endswith("pypi: not used")
    assert err == "visits=3 comments=1 cost=$0.000000\n"


def test_a_comment_is_still_written_when_every_source_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    respx_mock.route().respond(500)
    text = "It is widely used, though my searches failed."
    llm = FakeLLM(rules=[*_OTHERS_PASS, (_is_research, _RESEARCH), (_is_decide, _COMMENT_ON_POST), (_is_write, text)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    lines, err = _post_and_run(capsys)

    for call in _calls_of(llm, "Developer")[1:]:  # the decide and write calls
        for tool in ("github_search", "github_repo", "pypi"):
            assert f'{tool}, query "httpx": the tool failed' in call.messages[0].content
    assert "Developer [AI] commented: #1" in lines
    assert err.count("tool failed") == 3
    assert _show(capsys) == f"me:\nhttpx\n\n  #1 Developer [AI]:\n  {text}\n"


@pytest.mark.usefixtures("mock_github", "mock_pypi", "mock_arxiv", "mock_youtube")
def test_all_three_agents_research_with_their_own_tools_and_comment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    comments = {
        "Developer": "Developers use it: [httpx on PyPI](https://pypi.org/project/httpx/).",
        "Scientist": "Researchers write about it: [papers](https://arxiv.org/abs/2010.11929).",
        "Pollster": "The public watches it: [videos](https://www.youtube.com/results?search_query=httpx).",
    }
    plans = {
        "Scientist": ResearchPlan(speak=True, queries={"arxiv": ["vision transformer"]}, reason="Papers exist."),
        "Pollster": ResearchPlan(speak=True, queries={"youtube": ["rust programming"]}, reason="Videos exist."),
    }

    def later_look(request: LLMRequest) -> bool:
        """A decide call on the other agents' comments, which notify an agent that has commented on the post."""
        return request.prompt_id == "decide.v1" and "New to you: the post" not in request.messages[0].content

    llm = FakeLLM(
        rules=[
            (later_look, _NOTHING_NEW),
            (_is_research, _RESEARCH),
            *((_is(agent, "research.v1"), plan) for agent, plan in plans.items()),
            *((_is(agent, "decide.v1"), _COMMENT_ON_POST) for agent in comments),
            *((_is(agent, f"{agent.lower()}.v1"), text) for agent, text in comments.items()),
        ]
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    lines, err = _post_and_run(capsys)

    # Each agent researches once and writes with its own prompt, and its write call holds only its own tools' results.
    reports: dict[str, str] = {}
    for agent in comments:
        calls = _calls_of(llm, agent)
        assert [call.prompt_id for call in calls].count("research.v1") == 1
        [write] = [call for call in calls if call.prompt_id == f"{agent.lower()}.v1"]
        reports[agent] = write.messages[0].content.split("Tool results:")[1]
    assert "github.com" in reports["Developer"] and "arxiv.org" not in reports["Developer"]
    assert "arxiv.org" in reports["Scientist"] and "github.com" not in reports["Scientist"]
    assert "youtube.com" in reports["Pollster"] and "github.com" not in reports["Pollster"]
    assert "youtube.com" not in reports["Developer"] + reports["Scientist"]
    assert "arxiv.org" not in reports["Pollster"]

    # Each top-level comment notified the agents that had already commented, which then passed.
    commented = [line for line in lines if " commented: " in line]
    assert sorted(line.split(" [AI]")[0] for line in commented) == ["Developer", "Pollster", "Scientist"]
    assert all(" passed: " in line for line in lines if line not in commented)
    assert err.splitlines()[-1].endswith("comments=3 cost=$0.000000")
    shown = _show(capsys)
    for text in comments.values():
        assert text in shown
