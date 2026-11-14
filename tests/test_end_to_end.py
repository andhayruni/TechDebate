"""A complete `techdebate run` with no network: only the model and the HTTP responses are replaced."""

from collections.abc import Callable

import pytest
import respx

from techdebate import cli
from techdebate.config import Settings
from techdebate.llm import FakeLLM, LLMRequest
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import ResearchPlan, SpeakDecision

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
_COMMENT_ON_POST = SpeakDecision(reply_to=[0], reason="The download numbers are new.")
_NOTHING_NEW = SpeakDecision(reply_to=[], reason="The results only repeat the post.")
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
def _no_local_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """A local .env can't change the result. With a YouTube key every agent takes part, and arXiv's spacing between
    requests is off."""
    settings = Settings(_env_file=None, YOUTUBE_API_KEY="key", arxiv_request_interval_s=0)  # pyright: ignore[reportCallIssue]
    monkeypatch.setattr(cli, "get_settings", lambda: settings)


@pytest.mark.usefixtures("mock_github", "mock_pypi")
def test_the_developer_comments(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    llm = FakeLLM(
        rules=[*_OTHERS_PASS, (_is_research, _RESEARCH), (_is_decide, _COMMENT_ON_POST), (_is_write, _COMMENT)]
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    research, decide, write, *others = llm.calls
    assert (research.prompt_id, decide.prompt_id, write.prompt_id) == ("research.v1", "decide.v1", "developer.v1")
    assert [call.prompt_id for call in others] == ["research.v1", "research.v1"]  # the Scientist, then the Pollster
    assert research.messages[0].content == "Post: httpx"
    # The write call gets what the decide call saw, and the point the decide call agreed to.
    report = decide.messages[0].content.removeprefix("Post: httpx")
    assert write.messages[0].content == f"Post: httpx{report}\n\nYour point: {_COMMENT_ON_POST.reason}"
    assert report.startswith('\n\nTool results:\n\ngithub_search, query "httpx":\n')
    for url in (
        "https://github.com/search?q=httpx&type=repositories",  # github_search
        "https://github.com/fastapi/fastapi/commits",  # github_repo
        "https://pypi.org/project/httpx/",  # pypi
    ):
        assert url in report
    for call in (research, decide, write):
        assert "httpx" not in call.system
        assert "You are the Developer" in call.system

    out, err = capsys.readouterr()
    assert out == f"me:\nhttpx\n\n  Developer [AI]:\n  {_COMMENT}\n"
    assert err == "comments=1 cost=$0.000000\n"  # nothing else, so no tool failed and no role was left out


def test_every_agent_passes_at_the_research_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    llm = FakeLLM(rules=[*_OTHERS_PASS, (_is_research, _SKIP)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    assert [request.prompt_id for request in llm.calls] == ["research.v1"] * 3
    assert respx_mock.calls.call_count == 0
    out, err = capsys.readouterr()
    assert out == "me:\nhttpx\n\n  No comments.\n"
    assert err == "comments=0 cost=$0.000000\n"


@pytest.mark.usefixtures("mock_github", "mock_pypi")
def test_the_developer_researches_and_then_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    llm = FakeLLM(rules=[*_OTHERS_PASS, (_is_research, _RESEARCH), (_is_decide, _NOTHING_NEW)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    assert [request.prompt_id for request in llm.calls] == ["research.v1", "decide.v1", "research.v1", "research.v1"]
    assert respx_mock.calls.call_count > 0  # the searches ran
    out, err = capsys.readouterr()
    assert out == "me:\nhttpx\n\n  No comments.\n"
    assert err == "comments=0 cost=$0.000000\n"


@pytest.mark.usefixtures("mock_github")
def test_the_developer_leaves_out_a_tool_that_does_not_fit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = FakeLLM(
        rules=[*_OTHERS_PASS, (_is_research, _NO_PYPI), (_is_decide, _COMMENT_ON_POST), (_is_write, _COMMENT)]
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])  # a request to PyPI would have no mock, and fail the test

    assert llm.calls[1].messages[0].content.endswith("pypi: not used")
    assert capsys.readouterr().err == "comments=1 cost=$0.000000\n"


def test_a_comment_is_still_written_when_every_source_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    respx_mock.route().respond(500)
    text = "It is widely used, though my searches failed."
    llm = FakeLLM(rules=[*_OTHERS_PASS, (_is_research, _RESEARCH), (_is_decide, _COMMENT_ON_POST), (_is_write, text)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    for call in llm.calls[1:3]:  # the Developer's decide and write calls
        for tool in ("github_search", "github_repo", "pypi"):
            assert f'{tool}, query "httpx": the tool failed' in call.messages[0].content
    out, err = capsys.readouterr()
    assert out == f"me:\nhttpx\n\n  Developer [AI]:\n  {text}\n"
    assert err.count("tool failed") == 3


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
    llm = FakeLLM(
        rules=[
            (_is_research, _RESEARCH),
            *((_is(agent, "research.v1"), plan) for agent, plan in plans.items()),
            *((_is(agent, "decide.v1"), _COMMENT_ON_POST) for agent in comments),
            *((_is(agent, f"{agent.lower()}.v1"), text) for agent, text in comments.items()),
        ]
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    # One agent after another, in registry order, each writing with its own prompt.
    assert [call.prompt_id for call in llm.calls] == [
        *("research.v1", "decide.v1", "developer.v1"),
        *("research.v1", "decide.v1", "scientist.v1"),
        *("research.v1", "decide.v1", "pollster.v1"),
    ]
    # Each agent's write call holds only its own tools' results.
    developer, scientist, pollster = (llm.calls[i].messages[0].content for i in (2, 5, 8))
    assert "github.com" in developer and "arxiv.org" not in developer and "youtube.com" not in developer
    assert "arxiv.org" in scientist and "github.com" not in scientist and "youtube.com" not in scientist
    assert "youtube.com" in pollster and "github.com" not in pollster and "arxiv.org" not in pollster
    assert scientist.startswith('Post: httpx\n\nTool results:\n\narxiv, query "vision transformer":\n')
    assert pollster.startswith('Post: httpx\n\nTool results:\n\nyoutube, query "rust programming":\n')

    out, err = capsys.readouterr()
    assert out == (
        "me:\nhttpx\n\n"
        f"  Developer [AI]:\n  {comments['Developer']}\n\n"
        f"  Scientist [AI]:\n  {comments['Scientist']}\n\n"
        f"  Pollster [AI]:\n  {comments['Pollster']}\n"
    )
    assert err == "comments=3 cost=$0.000000\n"
