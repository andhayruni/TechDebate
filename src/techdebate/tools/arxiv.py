"""arXiv tool: papers with a phrase in their title, the best matches, and how many were submitted each year.

The API answers in Atom XML. It has no counts grouped by year, so the trend costs one request per year, and arXiv asks
for at most one request every 3 seconds: with the default 5 years, a run takes about 15 seconds. A count request needs
`max_results=1`, because `max_results=0` fails with HTTP 500.
"""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import urlencode
from xml.etree.ElementTree import Element

import httpx
from defusedxml import DefusedXmlException
from defusedxml.ElementTree import ParseError, fromstring
from pydantic import Field, HttpUrl

from techdebate.config import Settings
from techdebate.errors import ToolError
from techdebate.models import Evidence
from techdebate.tools.base import ToolInput

API = "https://export.arxiv.org/api/query"
WEB = "https://arxiv.org"

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
# The version suffix of an abstract URL ("2104.08500v4"), so the link always opens the latest version.
_VERSION = re.compile(r"v\d+$")
_AUTHORS_SHOWN = 3
_ABSTRACT_CHARS = 300


def _today() -> date:
    return datetime.now(UTC).date()


class _Throttle:
    """Spaces calls at least `interval` seconds apart. The lock makes concurrent callers wait their turn too."""

    def __init__(self, interval: float, sleep: Callable[[float], Awaitable[None]], clock: Callable[[], float]) -> None:
        self._interval = interval
        self._sleep = sleep
        self._clock = clock
        self._lock = asyncio.Lock()
        self._last: float | None = None

    async def wait(self) -> None:
        async with self._lock:
            if self._last is not None:
                delay = self._last + self._interval - self._clock()
                if delay > 0:
                    await self._sleep(delay)
            self._last = self._clock()


@dataclass(frozen=True)
class _Paper:
    title: str
    authors: list[str]
    submitted: str
    abstract: str
    url: str


@dataclass(frozen=True)
class _Feed:
    total: int
    papers: list[_Paper]


def _text(element: Element, path: str) -> str:
    """The text at `path`, with line breaks and runs of spaces collapsed (arXiv wraps long titles and abstracts)."""
    return " ".join((element.findtext(path, default="", namespaces=_NS)).split())


def _root(body: str) -> Element:
    try:
        return fromstring(body)
    except (ParseError, DefusedXmlException) as e:
        raise ToolError(f"arxiv: invalid response: {e}") from e


def _error(root: Element) -> str | None:
    """arXiv's own error message, which it sends as a feed with a single entry under /api/errors."""
    for entry in root.iterfind("atom:entry", _NS):
        if "/api/errors" in _text(entry, "atom:id"):
            return _text(entry, "atom:summary") or "unknown error"
    return None


def _parse(body: str) -> _Feed:
    root = _root(body)
    if (message := _error(root)) is not None:
        raise ToolError(f"arxiv: {message}")
    try:
        total = int(_text(root, "opensearch:totalResults"))
    except ValueError as e:
        raise ToolError("arxiv: invalid response: no total count") from e
    papers = [
        _Paper(
            title=_text(entry, "atom:title"),
            authors=[_text(author, "atom:name") for author in entry.iterfind("atom:author", _NS)],
            submitted=_text(entry, "atom:published")[:10],
            abstract=_text(entry, "atom:summary"),
            url=_VERSION.sub("", _text(entry, "atom:id")).replace("http://", "https://", 1),
        )
        for entry in root.iterfind("atom:entry", _NS)
    ]
    return _Feed(total, papers)


def _authors(names: list[str]) -> str:
    shown = ", ".join(names[:_AUTHORS_SHOWN])
    return f"{shown} et al." if len(names) > _AUTHORS_SHOWN else shown


def _snippet(text: str) -> str:
    """The start of an abstract: whole sentences up to about 300 characters, or the first sentence cut at a word."""
    if len(text) <= _ABSTRACT_CHARS:
        return text
    snippet = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(snippet) + len(sentence) + 1 > _ABSTRACT_CHARS:
            break
        snippet = f"{snippet} {sentence}".strip()
    if snippet:
        return snippet
    return text[: _ABSTRACT_CHARS + 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _search_page(phrase: str, newest_first: bool = False) -> HttpUrl:
    params = {"query": f'"{phrase}"', "searchtype": "title"}
    if newest_first:
        params["order"] = "-announced_date_first"
    return HttpUrl(f"{WEB}/search/?{urlencode(params)}")


class ArxivInput(ToolInput):
    query: str = Field(description="A phrase to find in paper titles")


class ArxivTool:
    name = "arxiv"
    description = "arXiv: papers with the phrase in their title, the best matches, and how many per year"
    input_model = ArxivInput

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: Settings,
        today: Callable[[], date] = _today,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._top_n = settings.arxiv_top_n
        self._trend_years = settings.arxiv_trend_years
        self._today = today
        self._throttle = _Throttle(settings.arxiv_request_interval_s, sleep, clock)

    async def run(self, input: ArxivInput) -> list[Evidence]:
        # Quotes would end the phrase early, so they go, along with extra whitespace.
        phrase = " ".join(input.query.replace('"', " ").split())
        if not phrase:
            raise ToolError("arxiv: empty query")

        feed = await self._get(
            {"search_query": f'ti:"{phrase}"', "sortBy": "relevance", "max_results": str(self._top_n)}
        )
        total = Evidence(
            source=self.name,
            query=input.query,
            url=_search_page(phrase),
            title=f'arXiv: "{phrase}" in titles',
            summary=f'{feed.total:,} papers on arXiv have "{phrase}" in the title',
            value=feed.total,
        )
        if feed.total == 0:
            return [total]

        evidence = [total]
        if self._trend_years:
            evidence.append(await self._trend(input.query, phrase))
        evidence += [self._paper(input.query, paper) for paper in feed.papers]
        return evidence

    async def _get(self, params: dict[str, str]) -> _Feed:
        await self._throttle.wait()
        try:
            response = await self._client.get(API, params=params)
        except httpx.HTTPError as e:
            raise ToolError(f"arxiv: request failed: {e}") from e
        if response.status_code in (429, 503):
            raise ToolError(f"arxiv: rate limit reached or service unavailable (HTTP {response.status_code})")
        if not response.is_success:
            try:
                message = _error(_root(response.text))
            except ToolError:
                message = None
            raise ToolError(f"arxiv: HTTP {response.status_code}" + (f": {message}" if message else ""))
        return _parse(response.text)

    async def _trend(self, query: str, phrase: str) -> Evidence:
        today = self._today()
        first = today.year - self._trend_years + 1
        parts: list[str] = []
        last_complete: int | None = None
        # One year at a time: the throttle spaces the requests anyway.
        for year in range(first, today.year + 1):
            end = today if year == today.year else date(year, 12, 31)
            feed = await self._get(
                {
                    "search_query": f'ti:"{phrase}" AND submittedDate:[{year}01010000 TO {end:%Y%m%d}2359]',
                    "max_results": "1",
                }
            )
            label = str(year)
            if year == today.year:
                label += f" (to {today:%m-%d})"
            else:
                last_complete = feed.total
            parts.append(f"{label}: {feed.total:,}")
        return Evidence(
            source=self.name,
            query=query,
            url=_search_page(phrase, newest_first=True),
            title=f'arXiv: "{phrase}" papers per year',
            summary=f'Papers with "{phrase}" in the title, by submission year: {"; ".join(parts)}.',
            value=last_complete,
        )

    def _paper(self, query: str, paper: _Paper) -> Evidence:
        byline = f"Submitted {paper.submitted}."
        if paper.authors:
            byline = f"By {_authors(paper.authors)}; submitted {paper.submitted}."
        return Evidence(
            source=self.name,
            query=query,
            url=HttpUrl(paper.url),
            title=paper.title,
            summary=f"{byline} {_snippet(paper.abstract)}".rstrip(),
        )
