"""GitHub tools: repository search by keywords, and the activity of one repository.

Without a token the search API allows 10 requests a minute (30 with one). github_repo makes two search calls per run,
three when it resolves keywords to a repository, so a run without a token uses a third of the minute's searches.
"""

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import httpx
from pydantic import Field, HttpUrl

from techdebate.config import Settings
from techdebate.errors import ToolError
from techdebate.models import Evidence
from techdebate.tools.base import ToolInput

API = "https://api.github.com"
WEB = "https://github.com"

_FULL_NAME = re.compile(r"^[\w.-]+/[\w.-]+$")


def _now() -> datetime:
    return datetime.now(UTC)


def _headers(settings: Settings) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    # An empty GITHUB_TOKEN= line in .env counts as no token.
    token = settings.github_token.get_secret_value() if settings.github_token else ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _get(
    client: httpx.AsyncClient,
    path: str,
    headers: dict[str, str],
    params: dict[str, str | int] | None = None,
    allow: tuple[int, ...] = (),
) -> httpx.Response:
    """GET an API path. Failures raise ToolError; a status in `allow` is returned for the caller to handle."""
    try:
        response = await client.get(f"{API}{path}", params=params, headers=headers)
    except httpx.HTTPError as e:
        raise ToolError(f"github: request failed for {path}: {e}") from e
    if response.is_success or response.status_code in allow:
        return response
    if response.status_code in (403, 429) and _is_rate_limited(response):
        hint = "" if "Authorization" in headers else " (set GITHUB_TOKEN for a higher limit)"
        raise ToolError(f"github: rate limit reached{hint}")
    raise ToolError(f"github: HTTP {response.status_code} for {path}")


def _is_rate_limited(response: httpx.Response) -> bool:
    if response.headers.get("x-ratelimit-remaining") == "0":
        return True
    try:
        message = str(response.json().get("message", ""))
    except ValueError:
        return False
    return "rate limit" in message.lower()


def _day(timestamp: str | None) -> str:
    """The date part of a GitHub timestamp ("2026-10-16T04:47:44Z" -> "2026-10-16")."""
    return timestamp[:10] if timestamp else "unknown"


def _sentence(description: str | None) -> str:
    """A repository description as a sentence, so the numbers after it read cleanly."""
    text = (description or "").strip()
    if not text:
        return "No description."
    return text if text[-1] in ".!?" else f"{text}."


def _web_url(path: str, **params: str) -> HttpUrl:
    return HttpUrl(str(httpx.URL(f"{WEB}{path}", params=params)))


class GitHubSearchInput(ToolInput):
    query: str = Field(description="Keywords to search repositories for")


class GitHubSearchTool:
    name = "github_search"
    description = "GitHub: how many repositories match, and the most-starred ones"
    input_model = GitHubSearchInput

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._headers = _headers(settings)
        self._top_n = settings.github_top_n

    async def run(self, input: GitHubSearchInput) -> list[Evidence]:
        params: dict[str, str | int] = {"q": input.query, "sort": "stars", "order": "desc", "per_page": self._top_n}
        data: dict[str, Any] = (await _get(self._client, "/search/repositories", self._headers, params)).json()

        total: int = data["total_count"]
        evidence = [
            Evidence(
                source=self.name,
                query=input.query,
                url=_web_url("/search", q=input.query, type="repositories"),
                title=f"GitHub repositories matching {input.query!r}",
                summary=f"{total:,} repositories match {input.query!r} on GitHub",
                value=total,
            )
        ]
        for repo in data["items"]:
            evidence.append(
                Evidence(
                    source=self.name,
                    query=input.query,
                    url=repo["html_url"],
                    title=repo["full_name"],
                    summary=(
                        f"{repo['stargazers_count']:,} stars, {repo['forks_count']:,} forks, "
                        f"language {repo['language'] or 'not set'}, created {_day(repo['created_at'])}, "
                        f"last push {_day(repo['pushed_at'])}. {_sentence(repo['description'])}"
                    ),
                    value=repo["stargazers_count"],
                )
            )
        return evidence


class GitHubRepoInput(ToolInput):
    query: str = Field(description="A repository as owner/name, or keywords resolved to the most-starred match")


class GitHubRepoTool:
    name = "github_repo"
    description = "GitHub: one repository's stars, latest release, and recent commits, merged PRs and closed issues"
    input_model = GitHubRepoInput

    def __init__(self, client: httpx.AsyncClient, settings: Settings, now: Callable[[], datetime] = _now) -> None:
        self._client = client
        self._headers = _headers(settings)
        self._days = settings.github_activity_days
        self._now = now

    async def run(self, input: GitHubRepoInput) -> list[Evidence]:
        full_name = await self._resolve(input.query)

        response = await _get(self._client, f"/repos/{full_name}", self._headers, allow=(404,))
        if response.status_code == 404:
            raise ToolError(f"github: unknown repository {full_name}")
        repo: dict[str, Any] = response.json()
        # Use the canonical name from here on: the input may differ in case, or name a renamed repository.
        full_name = repo["full_name"]
        since = (self._now() - timedelta(days=self._days)).date()

        info = self._info(input.query, repo)
        release, commits, prs, issues = await asyncio.gather(
            self._release(input.query, repo),
            self._commits(input.query, repo, since),
            self._search_count(input.query, repo, since, kind="pr"),
            self._search_count(input.query, repo, since, kind="issue"),
        )
        return [info, release, commits, prs, issues]

    async def _resolve(self, query: str) -> str:
        query = query.strip()
        if _FULL_NAME.match(query):
            return query
        params: dict[str, str | int] = {"q": query, "sort": "stars", "order": "desc", "per_page": 1}
        items: list[dict[str, Any]] = (await _get(self._client, "/search/repositories", self._headers, params)).json()[
            "items"
        ]
        if not items:
            raise ToolError(f"github: no repository matches {query!r}")
        return items[0]["full_name"]

    def _info(self, query: str, repo: dict[str, Any]) -> Evidence:
        license_: dict[str, Any] | None = repo.get("license")
        topics: list[str] = repo.get("topics") or []
        archived = " Archived." if repo.get("archived") else ""
        return Evidence(
            source=self.name,
            query=query,
            url=repo["html_url"],
            title=repo["full_name"],
            summary=(
                f"{_sentence(repo['description'])} "
                f"{repo['stargazers_count']:,} stars, {repo['forks_count']:,} forks, "
                f"{repo['subscribers_count']:,} watchers, {repo['open_issues_count']:,} open issues and PRs, "
                f"license {license_['spdx_id'] if license_ else 'not set'}, "
                f"topics {', '.join(topics) if topics else 'none'}, "
                f"created {_day(repo['created_at'])}, last push {_day(repo['pushed_at'])}.{archived}"
            ),
            value=repo["stargazers_count"],
        )

    async def _release(self, query: str, repo: dict[str, Any]) -> Evidence:
        name = repo["full_name"]
        response = await _get(self._client, f"/repos/{name}/releases/latest", self._headers, allow=(404,))
        if response.status_code == 404:
            return Evidence(
                source=self.name,
                query=query,
                url=_web_url(f"/{name}/releases"),
                title=f"{name}: releases",
                summary=f"{name} has no published releases on GitHub",
            )
        release: dict[str, Any] = response.json()
        label = release["name"] or release["tag_name"]
        return Evidence(
            source=self.name,
            query=query,
            url=release["html_url"],
            title=f"{name}: latest release",
            summary=f"Latest release {label} (tag {release['tag_name']}), published {_day(release['published_at'])}",
        )

    async def _commits(self, query: str, repo: dict[str, Any], since: date) -> Evidence:
        name = repo["full_name"]
        params: dict[str, str | int] = {"since": f"{since.isoformat()}T00:00:00Z", "per_page": 1}
        # An empty repository answers 409 rather than an empty list.
        response = await _get(self._client, f"/repos/{name}/commits", self._headers, params, allow=(409,))
        commits: list[dict[str, Any]] = [] if response.status_code == 409 else response.json()

        # With one commit per page, the page number of the last page is the number of commits.
        last = response.links.get("last")
        count = int(httpx.URL(last["url"]).params["page"]) if last else len(commits)
        latest = f", the latest on {_day(commits[0]['commit']['committer']['date'])}" if commits else ""
        return Evidence(
            source=self.name,
            query=query,
            url=_web_url(f"/{name}/commits"),
            title=f"{name}: commits in the last {self._days} days",
            summary=f"{count:,} commits to the default branch in the last {self._days} days{latest}",
            value=count,
        )

    async def _search_count(
        self, query: str, repo: dict[str, Any], since: date, kind: Literal["pr", "issue"]
    ) -> Evidence:
        name = repo["full_name"]
        if kind == "pr":
            filters, page, what = f"is:pr is:merged merged:>={since.isoformat()}", "pulls", "pull requests merged"
        else:
            filters, page, what = f"is:issue is:closed closed:>={since.isoformat()}", "issues", "issues closed"
        params: dict[str, str | int] = {"q": f"repo:{name} {filters}", "per_page": 1}
        total: int = (await _get(self._client, "/search/issues", self._headers, params)).json()["total_count"]
        return Evidence(
            source=self.name,
            query=query,
            url=_web_url(f"/{name}/{page}", q=filters),
            title=f"{name}: {what} in the last {self._days} days",
            summary=f"{total:,} {what} in the last {self._days} days (since {since.isoformat()})",
            value=total,
        )
