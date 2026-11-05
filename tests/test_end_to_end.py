"""A complete `techdebate run` with no network: only the model and the HTTP responses are replaced."""

import pytest
import respx

from techdebate import cli
from techdebate.config import Settings
from techdebate.llm import FakeLLM, LLMRequest
from techdebate.models import SpeakDecision

_SPEAK = SpeakDecision(
    speak=True,
    reply_to=None,
    queries={"github_search": ["httpx"], "github_repo": ["httpx"], "pypi": ["httpx"]},
    reason="Developers use this.",
)
_NO_PYPI = SpeakDecision(
    speak=True,
    reply_to=None,
    queries={"github_search": ["httpx"], "github_repo": ["httpx"], "pypi": []},
    reason="Developers use this, but it is not a package.",
)
_PASS = SpeakDecision(
    speak=False, reply_to=None, queries={"github_search": [], "github_repo": [], "pypi": []}, reason="Not my area."
)
_TURN = "Developers use it, with [heavy downloads](https://pypistats.org/packages/httpx)."


def _is_decide(request: LLMRequest) -> bool:
    return request.prompt_id == "decide.v1"


def _is_write(request: LLMRequest) -> bool:
    return request.prompt_id == "developer.v1"


@pytest.fixture(autouse=True)
def _no_local_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """A local .env can't change the result."""
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))  # pyright: ignore[reportCallIssue]


@pytest.mark.usefixtures("mock_github", "mock_pypi")
def test_the_developer_speaks(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    llm = FakeLLM(rules=[(_is_decide, _SPEAK), (_is_write, _TURN)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx", "--no-delay"])

    decide, write = llm.calls
    assert (decide.prompt_id, write.prompt_id) == ("decide.v1", "developer.v1")
    assert decide.messages[0].content == "Topic: httpx"
    content = write.messages[0].content
    assert content.startswith('Topic: httpx\n\nTool results:\n\ngithub_search, query "httpx":\n')
    for url in (
        "https://github.com/search?q=httpx&type=repositories",  # github_search
        "https://github.com/fastapi/fastapi/commits",  # github_repo
        "https://pypi.org/project/httpx/",  # pypi
    ):
        assert url in content
    assert "httpx" not in write.system
    assert "You are the Developer" in write.system

    out, err = capsys.readouterr()
    assert out == f"Developer:\n{_TURN}\n"
    assert err == "topic='httpx' turns=1 cost=$0.000000\n"  # nothing else, so no tool failed


def test_the_developer_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    llm = FakeLLM(rules=[(_is_decide, _PASS)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx", "--no-delay"])

    assert [request.prompt_id for request in llm.calls] == ["decide.v1"]
    assert respx_mock.calls.call_count == 0
    out, err = capsys.readouterr()
    assert out == "Nobody joined the discussion.\n"
    assert err == "topic='httpx' turns=0 cost=$0.000000\n"


@pytest.mark.usefixtures("mock_github")
def test_the_developer_leaves_out_a_tool_that_does_not_fit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = FakeLLM(rules=[(_is_decide, _NO_PYPI), (_is_write, _TURN)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx", "--no-delay"])  # a request to PyPI would have no mock, and fail the test

    assert llm.calls[1].messages[0].content.endswith("pypi: not used")
    assert capsys.readouterr().err == "topic='httpx' turns=1 cost=$0.000000\n"


def test_a_turn_is_still_written_when_every_source_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], respx_mock: respx.MockRouter
) -> None:
    respx_mock.route().respond(500)
    text = "From general knowledge, it is used, though my searches failed."
    llm = FakeLLM(rules=[(_is_decide, _SPEAK), (_is_write, text)])
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)

    cli.main(["run", "httpx", "--no-delay"])

    content = llm.calls[1].messages[0].content
    for tool in ("github_search", "github_repo", "pypi"):
        assert f'{tool}, query "httpx": the tool failed' in content
    out, err = capsys.readouterr()
    assert out == f"Developer:\n{text}\n"
    assert err.count("tool failed") == 3
