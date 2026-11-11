"""A complete `techdebate run` with no network: only the model and the HTTP responses are replaced."""

import pytest
import respx

from techdebate import cli
from techdebate.config import Settings
from techdebate.llm import FakeLLM, LLMRequest
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


def _is_research(request: LLMRequest) -> bool:
    return request.prompt_id == "research.v1"


def _is_decide(request: LLMRequest) -> bool:
    return request.prompt_id == "decide.v1"


def _is_write(request: LLMRequest) -> bool:
    return request.prompt_id == "developer.v1"


@pytest.fixture(autouse=True)
def _no_local_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """A local .env can't change the result."""
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))  # pyright: ignore[reportCallIssue]


@pytest.mark.usefixtures("mock_github", "mock_pypi")
def test_the_developer_comments(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    llm = FakeLLM(rules=[(_is_research, _RESEARCH), (_is_decide, _COMMENT_ON_POST), (_is_write, _COMMENT)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    research, decide, write = llm.calls
    assert (research.prompt_id, decide.prompt_id, write.prompt_id) == ("research.v1", "decide.v1", "developer.v1")
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
    assert err == "comments=1 cost=$0.000000\n"  # nothing else, so no tool failed


def test_the_developer_passes_at_the_research_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    llm = FakeLLM(rules=[(_is_research, _SKIP)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    assert [request.prompt_id for request in llm.calls] == ["research.v1"]
    assert respx_mock.calls.call_count == 0
    out, err = capsys.readouterr()
    assert out == "me:\nhttpx\n\n  No comments.\n"
    assert err == "comments=0 cost=$0.000000\n"


@pytest.mark.usefixtures("mock_github", "mock_pypi")
def test_the_developer_researches_and_then_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    llm = FakeLLM(rules=[(_is_research, _RESEARCH), (_is_decide, _NOTHING_NEW)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    assert [request.prompt_id for request in llm.calls] == ["research.v1", "decide.v1"]
    assert respx_mock.calls.call_count > 0  # the searches ran
    out, err = capsys.readouterr()
    assert out == "me:\nhttpx\n\n  No comments.\n"
    assert err == "comments=0 cost=$0.000000\n"


@pytest.mark.usefixtures("mock_github")
def test_the_developer_leaves_out_a_tool_that_does_not_fit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = FakeLLM(rules=[(_is_research, _NO_PYPI), (_is_decide, _COMMENT_ON_POST), (_is_write, _COMMENT)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])  # a request to PyPI would have no mock, and fail the test

    assert llm.calls[1].messages[0].content.endswith("pypi: not used")
    assert capsys.readouterr().err == "comments=1 cost=$0.000000\n"


def test_a_comment_is_still_written_when_every_source_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    respx_mock.route().respond(500)
    text = "It is widely used, though my searches failed."
    llm = FakeLLM(rules=[(_is_research, _RESEARCH), (_is_decide, _COMMENT_ON_POST), (_is_write, text)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx"])

    for call in llm.calls[1:]:  # the decide and the write calls
        for tool in ("github_search", "github_repo", "pypi"):
            assert f'{tool}, query "httpx": the tool failed' in call.messages[0].content
    out, err = capsys.readouterr()
    assert out == f"me:\nhttpx\n\n  Developer [AI]:\n  {text}\n"
    assert err.count("tool failed") == 3
