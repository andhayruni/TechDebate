from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from techdebate.config import Settings
from techdebate.errors import NotFoundError, ToolError
from techdebate.tools import GitHubRepoTool, GitHubSearchTool, Tool, make_http_client
from techdebate.tools.github import GitHubRepoInput, GitHubSearchInput

HOST = "api.github.com"
# 90 days before this is 2026-08-01, the window start the fixtures were recorded with.
NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
_COMMITS_PAGE = "https://api.github.com/repositories/724712/commits?since=2026-08-01T00%3A00%3A00Z&per_page=1&page="
COMMITS_LINK = f'<{_COMMITS_PAGE}2>; rel="next", <{_COMMITS_PAGE}9797>; rel="last"'
PRS_QUERY = "repo:rust-lang/rust is:pr is:merged merged:>=2026-08-01"
ISSUES_QUERY = "repo:rust-lang/rust is:issue is:closed closed:>=2026-08-01"

Fixture = Callable[[str], Any]


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with make_http_client(_settings()) as client:
        yield client


def _search(client: httpx.AsyncClient, **settings: Any) -> GitHubSearchTool:
    return GitHubSearchTool(client, _settings(**settings))


def _repo(client: httpx.AsyncClient, **settings: Any) -> GitHubRepoTool:
    return GitHubRepoTool(client, _settings(**settings), now=lambda: NOW)


def _mock_rust_repo(respx_mock: respx.MockRouter, load_fixture: Fixture) -> dict[str, respx.Route]:
    """Routes for every call github_repo makes for rust-lang/rust."""
    base = "/repos/rust-lang/rust"
    return {
        "info": respx_mock.get(host=HOST, path=base).respond(json=load_fixture("github_repo/repo_rust.json")),
        "release": respx_mock.get(host=HOST, path=f"{base}/releases/latest").respond(
            json=load_fixture("github_repo/release_rust.json")
        ),
        "commits": respx_mock.get(host=HOST, path=f"{base}/commits").respond(
            json=load_fixture("github_repo/commits_rust.json"), headers={"Link": COMMITS_LINK}
        ),
        "prs": respx_mock.get(host=HOST, path="/search/issues", params={"q": PRS_QUERY}).respond(
            json=load_fixture("github_repo/prs_rust.json")
        ),
        "issues": respx_mock.get(host=HOST, path="/search/issues", params={"q": ISSUES_QUERY}).respond(
            json=load_fixture("github_repo/issues_rust.json")
        ),
    }


def test_tools_satisfy_the_protocol(client: httpx.AsyncClient) -> None:
    search: Tool[GitHubSearchInput] = _search(client)  # pyright checks the assignments
    repo: Tool[GitHubRepoInput] = _repo(client)
    assert (search.name, repo.name) == ("github_search", "github_repo")


# github_search


async def test_search_returns_the_total_and_the_top_repos(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    route = respx_mock.get(host=HOST, path="/search/repositories").respond(json=load_fixture("github_search/rust.json"))

    evidence = await _search(client).run(GitHubSearchInput(query="rust"))

    params = route.calls.last.request.url.params
    assert dict(params) == {"q": "rust", "sort": "stars", "order": "desc", "per_page": "5"}

    total, *repos = evidence
    assert total.value == 683239
    assert str(total.url) == "https://github.com/search?q=rust&type=repositories"
    assert "683,239 repositories" in total.summary

    assert [(r.title, r.value) for r in repos] == [
        ("ultraworkers/claw-code", 194977),
        ("farion1231/cc-switch", 150331),
        ("krahets/hello-algo", 131749),
        ("rustdesk/rustdesk", 127574),
        ("rust-lang/rust", 120832),
    ]
    rust = repos[-1]
    assert str(rust.url) == "https://github.com/rust-lang/rust"
    assert "120,832 stars, 17,989 forks, language Rust, created 2010-06-16" in rust.summary
    assert {(e.source, e.query) for e in evidence} == {("github_search", "rust")}


async def test_search_takes_the_number_of_repos_from_settings(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    route = respx_mock.get(host=HOST, path="/search/repositories").respond(json=load_fixture("github_search/rust.json"))

    await _search(client, github_top_n=3).run(GitHubSearchInput(query="rust"))

    assert route.calls.last.request.url.params["per_page"] == "3"


async def test_search_with_no_matches_returns_only_the_zero_count(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    respx_mock.get(host=HOST, path="/search/repositories").respond(json=load_fixture("github_search/empty.json"))

    evidence = await _search(client).run(GitHubSearchInput(query="zzqxvbnmkplw"))

    assert [(e.value, e.summary) for e in evidence] == [(0, "0 repositories match 'zzqxvbnmkplw' on GitHub")]


async def test_token_is_sent_when_set(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    route = respx_mock.get(host=HOST, path="/search/repositories").respond(
        json=load_fixture("github_search/empty.json")
    )

    await _search(client, GITHUB_TOKEN="secret").run(GitHubSearchInput(query="x"))
    assert route.calls.last.request.headers["Authorization"] == "Bearer secret"
    assert route.calls.last.request.headers["X-GitHub-Api-Version"] == "2022-11-28"

    await _search(client).run(GitHubSearchInput(query="x"))
    assert "Authorization" not in route.calls.last.request.headers

    await _search(client, GITHUB_TOKEN="").run(GitHubSearchInput(query="x"))
    assert "Authorization" not in route.calls.last.request.headers


# Failures


async def test_rate_limit_raises_tool_error_with_a_token_hint(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(host=HOST, path="/search/repositories").respond(
        403, json={"message": "API rate limit exceeded for 1.2.3.4."}, headers={"x-ratelimit-remaining": "0"}
    )

    with pytest.raises(ToolError, match=r"rate limit reached \(set GITHUB_TOKEN"):
        await _search(client).run(GitHubSearchInput(query="rust"))


async def test_secondary_rate_limit_is_detected_from_the_message(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(host=HOST, path="/search/repositories").respond(
        403, json={"message": "You have exceeded a secondary rate limit."}
    )

    with pytest.raises(ToolError, match=r"^github: rate limit reached$"):
        await _search(client, GITHUB_TOKEN="secret").run(GitHubSearchInput(query="rust"))


async def test_server_error_raises_tool_error(client: httpx.AsyncClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host=HOST, path="/search/repositories").respond(500)

    with pytest.raises(ToolError, match="HTTP 500 for /search/repositories"):
        await _search(client).run(GitHubSearchInput(query="rust"))


async def test_connection_error_raises_tool_error(client: httpx.AsyncClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host=HOST, path="/search/repositories").mock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(ToolError, match="request failed for /search/repositories: boom"):
        await _search(client).run(GitHubSearchInput(query="rust"))


# github_repo


async def test_repo_reports_info_release_and_activity(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_rust_repo(respx_mock, load_fixture)

    info, release, commits, prs, issues = await _repo(client).run(GitHubRepoInput(query="rust-lang/rust"))

    assert routes["commits"].calls.last.request.url.params["since"] == "2026-08-01T00:00:00Z"
    assert routes["prs"].called and routes["issues"].called

    assert (info.title, info.value, str(info.url)) == ("rust-lang/rust", 120832, "https://github.com/rust-lang/rust")
    assert info.summary == (
        "Empowering everyone to build reliable and efficient software. "
        "120,832 stars, 17,989 forks, 1,554 watchers, 12,644 open issues and PRs, license Apache-2.0, "
        "topics compiler, language, rust, created 2010-06-16, last push 2026-10-30."
    )

    assert str(release.url) == "https://github.com/rust-lang/rust/releases/tag/1.99.1"
    assert release.summary == "Latest release Rust 1.99.1 (tag 1.99.1), published 2026-10-16"

    assert commits.value == 9797
    assert str(commits.url) == "https://github.com/rust-lang/rust/commits"
    assert commits.summary == "9,797 commits to the default branch in the last 90 days, the latest on 2026-10-29"

    assert prs.value == 2588
    assert str(prs.url) == "https://github.com/rust-lang/rust/pulls?q=is%3Apr+is%3Amerged+merged%3A%3E%3D2026-08-01"
    assert issues.value == 1415
    assert str(issues.url) == (
        "https://github.com/rust-lang/rust/issues?q=is%3Aissue+is%3Aclosed+closed%3A%3E%3D2026-08-01"
    )

    evidence = [info, release, commits, prs, issues]
    assert {(e.source, e.query) for e in evidence} == {("github_repo", "rust-lang/rust")}
    assert all(e.url.host == "github.com" for e in evidence)


async def test_repo_without_releases_says_so(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_rust_repo(respx_mock, load_fixture)
    routes["release"].respond(404, json=load_fixture("github_repo/not_found.json"))

    _, release, *_ = await _repo(client).run(GitHubRepoInput(query="rust-lang/rust"))

    assert release.summary == "rust-lang/rust has no published releases on GitHub"
    assert str(release.url) == "https://github.com/rust-lang/rust/releases"


async def test_repo_without_recent_commits_counts_zero(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_rust_repo(respx_mock, load_fixture)
    routes["commits"].respond(json=[])

    _, _, commits, *_ = await _repo(client).run(GitHubRepoInput(query="rust-lang/rust"))

    assert commits.value == 0
    assert commits.summary == "0 commits to the default branch in the last 90 days"


async def test_empty_repo_counts_zero_commits(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_rust_repo(respx_mock, load_fixture)
    routes["commits"].respond(409, json={"message": "Git Repository is empty."})

    _, _, commits, *_ = await _repo(client).run(GitHubRepoInput(query="rust-lang/rust"))

    assert commits.value == 0


async def test_window_comes_from_settings(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_rust_repo(respx_mock, load_fixture)
    prs_30 = respx_mock.get(
        host=HOST, path="/search/issues", params={"q": "repo:rust-lang/rust is:pr is:merged merged:>=2026-09-30"}
    ).respond(json={"total_count": 0})
    issues_30 = respx_mock.get(
        host=HOST, path="/search/issues", params={"q": "repo:rust-lang/rust is:issue is:closed closed:>=2026-09-30"}
    ).respond(json={"total_count": 0})

    *_, prs, _ = await _repo(client, github_activity_days=30).run(GitHubRepoInput(query="rust-lang/rust"))

    assert routes["commits"].calls.last.request.url.params["since"] == "2026-09-30T00:00:00Z"
    assert prs_30.called and issues_30.called
    assert prs.title == "rust-lang/rust: pull requests merged in the last 30 days"


async def test_unknown_repo_raises_tool_error(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    respx_mock.get(host=HOST, path="/repos/nope/missing").respond(404, json=load_fixture("github_repo/not_found.json"))

    with pytest.raises(NotFoundError, match="unknown repository nope/missing"):
        await _repo(client).run(GitHubRepoInput(query="nope/missing"))


async def test_keywords_resolve_to_the_top_search_hit(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    search = respx_mock.get(host=HOST, path="/search/repositories").respond(
        json=load_fixture("github_repo/search_top_hit.json")
    )
    # Only the repo info matters here; the other calls answer with empty data.
    fastapi = respx_mock.get(host=HOST, path="/repos/fastapi/fastapi").respond(
        json={**load_fixture("github_repo/repo_rust.json"), "full_name": "fastapi/fastapi", "description": "Fast API"}
    )
    respx_mock.get(host=HOST, path="/repos/fastapi/fastapi/commits").respond(json=[])
    respx_mock.get(host=HOST, path="/repos/fastapi/fastapi/releases/latest").respond(404)
    respx_mock.get(host=HOST, path="/search/issues").respond(json={"total_count": 0})

    info, *_ = await _repo(client).run(GitHubRepoInput(query="fastapi"))

    assert dict(search.calls.last.request.url.params) == {
        "q": "fastapi",
        "sort": "stars",
        "order": "desc",
        "per_page": "1",
    }
    assert fastapi.called
    assert info.title == "fastapi/fastapi"
    assert info.summary.startswith("Fast API. 120,832 stars")  # a period is added before the numbers
    assert info.query == "fastapi"


async def test_keywords_without_a_match_raise_tool_error(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    respx_mock.get(host=HOST, path="/search/repositories").respond(json=load_fixture("github_search/empty.json"))

    with pytest.raises(NotFoundError, match="no repository matches 'zzqxvbnmkplw'"):
        await _repo(client).run(GitHubRepoInput(query="zzqxvbnmkplw"))
