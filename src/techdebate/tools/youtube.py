"""YouTube tool: how many videos match, how many are recent, and the views, likes and comments of the top videos.

The match counts are YouTube's rough estimates, and they stop at 1,000,000: a popular topic reports that number both
overall and for the last 12 months, so only counts below it can show growth. YouTube keeps no history of views or likes.

Each run costs about 201 of the 10,000 free daily quota units (100 per search, 1 for the statistics). The key goes in
a header rather than the `key` URL parameter, so it stays out of URLs, logs and error messages.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import Field, HttpUrl

from techdebate.config import Settings
from techdebate.errors import ToolError
from techdebate.models import Evidence
from techdebate.tools.base import ToolInput

API = "https://www.googleapis.com/youtube/v3"
WEB = "https://www.youtube.com"

_ESTIMATE_CAP = 1_000_000
_RECENT_DAYS = 365


def _today() -> date:
    return datetime.now(UTC).date()


def _error_message(response: httpx.Response) -> str:
    """What went wrong, from the error body the API sends, which may not be JSON."""
    status = f"HTTP {response.status_code}"
    try:
        error: dict[str, Any] = response.json()["error"]
        reasons = [item["reason"] for item in error.get("errors", [])]
        message: str = error["message"]
    except (ValueError, KeyError, TypeError):
        return status
    if "quotaExceeded" in reasons:
        return "daily quota used up (quotaExceeded)"
    return f"{status}: {message}"


def _estimate(total: int) -> str:
    return f"at least {total:,}" if total >= _ESTIMATE_CAP else f"about {total:,}"


@dataclass(frozen=True)
class _Video:
    id: str
    title: str
    channel: str
    published: str
    # None when YouTube leaves the number out: the owner hid the likes or turned the comments off.
    views: int | None
    likes: int | None
    comments: int | None


def _number(statistics: dict[str, str], name: str) -> int | None:
    value = statistics.get(name)
    return int(value) if value is not None else None


def _video(item: dict[str, Any]) -> _Video:
    snippet: dict[str, Any] = item["snippet"]
    statistics: dict[str, str] = item.get("statistics", {})
    return _Video(
        id=item["id"],
        title=snippet["title"],
        channel=snippet["channelTitle"],
        published=snippet["publishedAt"][:10],
        views=_number(statistics, "viewCount"),
        likes=_number(statistics, "likeCount"),
        comments=_number(statistics, "commentCount"),
    )


def _sum(values: list[int | None], label: str, missing: str) -> str:
    """A sum over the top videos that says how many left the number out, instead of counting them as zero."""
    known = [value for value in values if value is not None]
    if not known:
        return f"{label} {missing} on all {len(values)}"
    if len(known) == len(values):
        return f"{sum(known):,} {label}"
    return f"{sum(known):,} {label} ({missing} on {len(values) - len(known)} of {len(values)})"


class YouTubeInput(ToolInput):
    query: str = Field(description="YouTube search terms")


class YouTubeTool:
    name = "youtube"
    description = "YouTube: how many videos match, how many are recent, and the views and likes of the top ones"
    input_model = YouTubeInput

    def __init__(
        self, client: httpx.AsyncClient, api_key: str, settings: Settings, today: Callable[[], date] = _today
    ) -> None:
        self._client = client
        self._headers = {"X-goog-api-key": api_key}
        self._top_n = settings.youtube_top_n
        self._today = today

    async def run(self, input: YouTubeInput) -> list[Evidence]:
        query = input.query.strip()
        if not query:
            raise ToolError("youtube: empty query")
        page = HttpUrl(f"{WEB}/results?{urlencode({'search_query': query})}")

        search = await self._search(query, order="relevance", maxResults=str(self._top_n))
        total = self._total(input.query, query, page, search["pageInfo"]["totalResults"])
        ids: list[str] = [item["id"]["videoId"] for item in search["items"]]
        # An empty search spends no more quota.
        if not ids:
            return [total]

        since = self._today() - timedelta(days=_RECENT_DAYS)
        recent, statistics = await asyncio.gather(
            self._search(query, publishedAfter=f"{since:%Y-%m-%d}T00:00:00Z", maxResults="1"),
            self._get("videos", {"part": "statistics,snippet", "id": ",".join(ids)}),
        )
        by_id = {video.id: video for video in map(_video, statistics["items"])}
        # In relevance order. A video deleted or made private since the search is missing from the statistics.
        videos = [by_id[id] for id in ids if id in by_id]

        evidence = [total, self._recent(input.query, query, page, recent["pageInfo"]["totalResults"], since)]
        if videos:
            evidence.append(self._top(input.query, query, page, videos))
        evidence += [self._single(input.query, video) for video in videos]
        return evidence

    async def _search(self, query: str, **params: str) -> dict[str, Any]:
        return await self._get("search", {"part": "id", "type": "video", "q": query, **params})

    async def _get(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            response = await self._client.get(f"{API}/{endpoint}", params=params, headers=self._headers)
        except httpx.HTTPError as e:
            raise ToolError(f"youtube: request failed: {e}") from e
        if not response.is_success:
            raise ToolError(f"youtube: {_error_message(response)}")
        try:
            return response.json()
        except ValueError as e:
            raise ToolError(f"youtube: invalid response: {e}") from e

    def _total(self, query: str, phrase: str, page: HttpUrl, total: int) -> Evidence:
        return Evidence(
            source=self.name,
            query=query,
            url=page,
            title=f'YouTube: videos on "{phrase}"',
            summary=f'YouTube estimates {_estimate(total)} videos match "{phrase}"',
            value=total,
        )

    def _recent(self, query: str, phrase: str, page: HttpUrl, total: int, since: date) -> Evidence:
        summary = f'YouTube estimates {_estimate(total)} videos matching "{phrase}" were published since {since}'
        if total >= _ESTIMATE_CAP:
            summary += f". YouTube's estimates stop at {_ESTIMATE_CAP:,}, so this can't show if interest is growing"
        return Evidence(
            source=self.name,
            query=query,
            url=page,
            title=f'YouTube: recent videos on "{phrase}"',
            summary=summary,
            value=total,
        )

    def _top(self, query: str, phrase: str, page: HttpUrl, videos: list[_Video]) -> Evidence:
        views = [video.views for video in videos]
        parts = [
            _sum(views, "views", "hidden"),
            _sum([video.likes for video in videos], "likes", "hidden"),
            _sum([video.comments for video in videos], "comments", "disabled"),
        ]
        known_views = [count for count in views if count is not None]
        return Evidence(
            source=self.name,
            query=query,
            url=page,
            title=f'YouTube: top {len(videos)} videos on "{phrase}"',
            summary=f"Summed over the top {len(videos)} videos, as of today: {', '.join(parts)}",
            value=sum(known_views) if known_views else None,
        )

    def _single(self, query: str, video: _Video) -> Evidence:
        views = f"{video.views:,} views" if video.views is not None else "views hidden"
        likes = f"{video.likes:,} likes" if video.likes is not None else "likes hidden"
        comments = f"{video.comments:,} comments" if video.comments is not None else "comments disabled"
        return Evidence(
            source=self.name,
            query=query,
            url=HttpUrl(f"{WEB}/watch?v={video.id}"),
            title=video.title,
            summary=f"By {video.channel}; published {video.published}. {views}, {likes}, {comments}.",
            value=video.views,
        )
