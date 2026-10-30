"""PyPI tool: a package's release history, and its downloads from pypistats.org.

pypistats keeps about 180 days of daily downloads, so that is all the history this tool sees. It only accepts
lowercase package names, and allows about 60 requests a minute.
"""

import asyncio
import calendar
import re
from datetime import date
from typing import Any

import httpx
from pydantic import Field, HttpUrl

from techdebate.errors import ToolError
from techdebate.models import Evidence
from techdebate.tools.base import ToolInput

PYPI = "https://pypi.org"
PYPISTATS = "https://pypistats.org"

# A valid project name (PEP 508), so a topic like "AI in general" never ends up in a URL path.
_NAME = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")


async def _get(client: httpx.AsyncClient, url: str, params: dict[str, str] | None = None) -> httpx.Response:
    """GET a URL. Transport failures, rate limits and error statuses other than 404 raise ToolError."""
    try:
        response = await client.get(url, params=params)
    except httpx.HTTPError as e:
        raise ToolError(f"pypi: request failed for {url}: {e}") from e
    if response.is_success or response.status_code == 404:
        return response
    if response.status_code == 429:
        raise ToolError(f"pypi: rate limit reached for {httpx.URL(url).host}")
    raise ToolError(f"pypi: HTTP {response.status_code} for {url}")


def _day(timestamp: str) -> str:
    """The date part of an ISO timestamp ("2024-12-06T15:37:21.509172Z" -> "2024-12-06")."""
    return timestamp[:10]


def _sentence(text: str | None) -> str:
    text = (text or "").strip()
    if not text:
        return "No summary."
    return text if text[-1] in ".!?" else f"{text}."


def _monthly(rows: list[dict[str, Any]]) -> tuple[list[str], int, date, date]:
    """Daily rows summed per month, as "2026-05: 123,456" parts with partial months marked, plus the total and range."""
    days = sorted((date.fromisoformat(row["date"]), int(row["downloads"])) for row in rows)
    first, last = days[0][0], days[-1][0]
    months: dict[tuple[int, int], int] = {}
    for day, downloads in days:
        months[(day.year, day.month)] = months.get((day.year, day.month), 0) + downloads

    parts: list[str] = []
    for (year, month), downloads in months.items():
        label = f"{year}-{month:02d}"
        if (year, month) == (first.year, first.month) and first.day > 1:
            label += f" (partial, from {first:%m-%d})"
        if (year, month) == (last.year, last.month) and last.day < calendar.monthrange(year, month)[1]:
            label += f" (partial, to {last:%m-%d})"
        parts.append(f"{label}: {downloads:,}")
    return parts, sum(months.values()), first, last


class PyPIInput(ToolInput):
    query: str = Field(description="A PyPI package name")


class PyPITool:
    name = "pypi"
    description = "PyPI: a package's releases and its downloads over the last ~6 months"
    input_model = PyPIInput

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def run(self, input: PyPIInput) -> list[Evidence]:
        query = input.query.strip()
        if not _NAME.match(query):
            raise ToolError(f"pypi: not a package name: {query!r}")

        response = await _get(self._client, f"{PYPI}/pypi/{query}/json")
        if response.status_code == 404:
            raise ToolError(f"pypi: unknown package {query}")
        package: dict[str, Any] = response.json()
        # The canonical name from here on; pypistats wants it in lowercase.
        name: str = package["info"]["name"]
        stats = f"{PYPISTATS}/api/packages/{name.lower()}"
        page = HttpUrl(f"{PYPISTATS}/packages/{name.lower()}")

        info = self._info(input.query, package)
        recent, overall = await asyncio.gather(
            _get(self._client, f"{stats}/recent"),
            _get(self._client, f"{stats}/overall", params={"mirrors": "false"}),
        )
        if recent.status_code == 404 or overall.status_code == 404:
            return [
                info,
                Evidence(
                    source=self.name,
                    query=input.query,
                    url=page,
                    title=f"{name}: downloads",
                    summary=f"pypistats.org has no download data for {name} yet",
                ),
            ]
        return [
            info,
            self._recent(input.query, name, page, recent.json()["data"]),
            self._overall(input.query, name, page, overall.json()["data"]),
        ]

    def _info(self, query: str, package: dict[str, Any]) -> Evidence:
        info: dict[str, Any] = package["info"]
        releases: dict[str, list[dict[str, Any]]] = package["releases"]
        # A version counts as released when it has at least one file that isn't yanked.
        count = sum(1 for files in releases.values() if any(not f["yanked"] for f in files))
        uploads = [f["upload_time_iso_8601"] for files in releases.values() for f in files]
        latest: list[dict[str, Any]] = package["urls"]

        parts = [f"Latest version {info['version']}"]
        if latest:
            parts[0] += f", released {_day(min(f['upload_time_iso_8601'] for f in latest))}"
        if uploads:
            parts.append(f"first release {_day(min(uploads))}")
        parts.append(f"{count:,} releases")
        return Evidence(
            source=self.name,
            query=query,
            url=HttpUrl(f"{PYPI}/project/{info['name']}/"),
            title=info["name"],
            summary=f"{_sentence(info['summary'])} {'; '.join(parts)}.",
        )

    def _recent(self, query: str, name: str, page: HttpUrl, data: dict[str, Any]) -> Evidence:
        return Evidence(
            source=self.name,
            query=query,
            url=page,
            title=f"{name}: recent downloads",
            summary=(
                f"{data['last_day']:,} downloads in the last day, {data['last_week']:,} in the last week, "
                f"{data['last_month']:,} in the last month (without mirrors)"
            ),
            value=data["last_month"],
        )

    def _overall(self, query: str, name: str, page: HttpUrl, rows: list[dict[str, Any]]) -> Evidence:
        title = f"{name}: monthly downloads"
        if not rows:
            return Evidence(
                source=self.name,
                query=query,
                url=page,
                title=title,
                summary=f"pypistats.org has no daily download data for {name}",
            )
        parts, total, first, last = _monthly(rows)
        days = (last - first).days + 1
        return Evidence(
            source=self.name,
            query=query,
            url=page,
            title=title,
            summary=(
                f"Monthly downloads without mirrors: {'; '.join(parts)}. "
                f"Total {total:,} over {days} days ({first} to {last}). pypistats.org keeps about 6 months."
            ),
            value=total,
        )
