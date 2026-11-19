import copy
import json
from collections.abc import AsyncIterator, Callable
from datetime import date
from typing import Any

import httpx
import pytest
import respx

from techdebate.config import Settings
from techdebate.errors import ToolError
from techdebate.tools import Tool, YouTubeTool, make_http_client
from techdebate.tools.youtube import YouTubeInput

SEARCH = "https://www.googleapis.com/youtube/v3/search"
VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
KEY = "secret-key"
TODAY = date(2026, 10, 30)
QUERY = "rust programming"
IDS = ["5C_HPTJg5ek", "tS1PFlTmpuU", "0y6RKiIk6cs"]

Fixture = Callable[[str], Any]


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with make_http_client(_settings()) as client:
        yield client


def _tool(client: httpx.AsyncClient, **settings: Any) -> YouTubeTool:
    return YouTubeTool(client, KEY, _settings(**settings), today=lambda: TODAY)


def _with_total(body: dict[str, Any], total: int) -> dict[str, Any]:
    body = copy.deepcopy(body)
    body["pageInfo"]["totalResults"] = total
    return body


def _mock(
    respx_mock: respx.MockRouter,
    load: Fixture,
    total: int = 5400,
    recent: int = 1200,
    videos: dict[str, Any] | None = None,
) -> tuple[respx.Route, respx.Route]:
    """The recorded responses, with totals below YouTube's cap by default (the recorded ones are at it)."""

    def search(request: httpx.Request) -> httpx.Response:
        if "publishedAfter" in request.url.params:
            return httpx.Response(200, json=_with_total(load("youtube/recent_rust.json"), recent))
        return httpx.Response(200, json=_with_total(load("youtube/search_rust.json"), total))

    search_route = respx_mock.get(SEARCH).mock(side_effect=search)
    videos_route = respx_mock.get(VIDEOS).mock(
        return_value=httpx.Response(200, json=videos or load("youtube/videos_rust.json"))
    )
    return search_route, videos_route


def _requests(route: respx.Route) -> list[httpx.Request]:
    return [call.request for call in route.calls]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


def _edited_videos(load: Fixture, index: int, *removed: str) -> dict[str, Any]:
    """The recorded statistics, with some numbers left out of one video, as YouTube does when they are hidden."""
    body = copy.deepcopy(load("youtube/videos_rust.json"))
    for name in removed:
        del body["items"][index]["statistics"][name]
    return body


def test_tool_satisfies_the_protocol(client: httpx.AsyncClient) -> None:
    tool: Tool[YouTubeInput] = _tool(client)
    assert tool.name == "youtube"
    assert tool.input_model is YouTubeInput


async def test_returns_the_counts_the_sums_and_one_evidence_per_video(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    _mock(respx_mock, load_fixture)

    evidence = await _tool(client).run(YouTubeInput(query=QUERY))

    assert [e.source for e in evidence] == ["youtube"] * 6
    assert {e.query for e in evidence} == {QUERY}
    total, recent, top, first, *_ = evidence
    assert total.value == 5400
    assert total.summary == 'YouTube estimates about 5,400 videos match "rust programming"'
    assert str(total.url) == "https://www.youtube.com/results?search_query=rust+programming"
    assert recent.value == 1200
    assert recent.summary == (
        'YouTube estimates about 1,200 videos matching "rust programming" were published since 2025-10-30'
    )
    assert top.value == 2635299 + 50556 + 508681
    assert top.summary == "Summed over the top 3 videos, as of today: 3,194,536 views, 100,029 likes, 2,398 comments"
    assert first.title == "Rust in 100 Seconds"
    assert first.value == 2635299
    assert first.summary == "By Fireship; published 2021-10-12. 2,635,299 views, 86,035 likes, 1,628 comments."
    assert str(first.url) == "https://www.youtube.com/watch?v=5C_HPTJg5ek"


async def test_sends_the_expected_requests_with_the_key_only_in_a_header(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    search, videos = _mock(respx_mock, load_fixture)

    await _tool(client, youtube_top_n=3).run(YouTubeInput(query=QUERY))

    assert search.call_count == 2
    params = [dict(request.url.params) for request in _requests(search)]
    assert params[0] == {"part": "id", "type": "video", "q": QUERY, "order": "relevance", "maxResults": "3"}
    assert params[1] == {
        "part": "id",
        "type": "video",
        "q": QUERY,
        "publishedAfter": "2025-10-30T00:00:00Z",
        "maxResults": "1",
    }
    assert dict(_requests(videos)[0].url.params) == {"part": "statistics,snippet", "id": ",".join(IDS)}
    for request in [*_requests(search), *_requests(videos)]:
        assert request.headers["X-goog-api-key"] == KEY
        assert KEY not in str(request.url)


async def test_the_recent_window_survives_a_leap_day(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    search, _ = _mock(respx_mock, load_fixture)
    tool = YouTubeTool(client, KEY, _settings(), today=lambda: date(2028, 2, 29))

    await tool.run(YouTubeInput(query=QUERY))

    assert _requests(search)[1].url.params["publishedAfter"] == "2027-03-01T00:00:00Z"


async def test_counts_at_the_cap_say_at_least_and_that_they_cannot_show_growth(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    _mock(respx_mock, load_fixture, total=1_000_000, recent=1_000_000)

    total, recent, *_ = await _tool(client).run(YouTubeInput(query=QUERY))

    assert "at least 1,000,000 videos" in total.summary
    assert "at least 1,000,000 videos" in recent.summary
    assert "can't show if interest is growing" in recent.summary


async def test_no_videos_gives_one_evidence_and_spends_no_more_quota(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    search = respx_mock.get(SEARCH).mock(return_value=httpx.Response(200, json=load_fixture("youtube/empty.json")))
    videos = respx_mock.get(VIDEOS)

    evidence = await _tool(client).run(YouTubeInput(query="nothing like this"))

    assert [e.value for e in evidence] == [0]
    assert search.call_count == 1
    assert not videos.called


async def test_hidden_likes_and_disabled_comments_are_reported_not_counted_as_zero(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    _mock(respx_mock, load_fixture, videos=_edited_videos(load_fixture, 0, "likeCount", "commentCount"))

    _, _, top, first, *_ = await _tool(client).run(YouTubeInput(query=QUERY))

    assert first.summary == "By Fireship; published 2021-10-12. 2,635,299 views, likes hidden, comments disabled."
    assert "13,994 likes (hidden on 1 of 3)" in top.summary
    assert "770 comments (disabled on 1 of 3)" in top.summary


async def test_numbers_hidden_on_every_video_are_not_summed_to_zero(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    body = load_fixture("youtube/videos_rust.json")
    for item in body["items"]:
        item["statistics"] = {}
    _mock(respx_mock, load_fixture, videos=body)

    _, _, top, first, *_ = await _tool(client).run(YouTubeInput(query=QUERY))

    assert top.value is None
    assert top.summary.endswith("views hidden on all 3, likes hidden on all 3, comments disabled on all 3")
    assert first.value is None
    assert "views hidden" in first.summary


async def test_a_video_missing_from_the_statistics_is_skipped_and_the_order_kept(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    body = load_fixture("youtube/videos_rust.json")
    del body["items"][1]
    body["items"].reverse()
    _mock(respx_mock, load_fixture, videos=body)

    evidence = await _tool(client).run(YouTubeInput(query=QUERY))

    assert [str(e.url).rsplit("=", 1)[1] for e in evidence[3:]] == [IDS[0], IDS[2]]
    assert "top 2 videos" in evidence[2].summary


async def test_no_statistics_at_all_still_returns_the_counts(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    _mock(respx_mock, load_fixture, videos={"items": []})

    evidence = await _tool(client).run(YouTubeInput(query=QUERY))

    assert [e.value for e in evidence] == [5400, 1200]


@pytest.mark.parametrize("query", ["", "   "])
async def test_empty_query_raises_before_any_request(client: httpx.AsyncClient, query: str) -> None:
    with pytest.raises(ToolError, match="empty query"):
        await _tool(client).run(YouTubeInput(query=query))


async def test_used_up_quota_raises_a_clear_error(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    body = load_fixture("youtube/quota_exceeded.json")
    respx_mock.get(SEARCH).mock(return_value=httpx.Response(403, json=body))

    with pytest.raises(ToolError, match=r"daily quota used up \(quotaExceeded\)"):
        await _tool(client).run(YouTubeInput(query=QUERY))


async def test_invalid_key_raises_with_youtubes_message_and_not_the_key(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    body = load_fixture("youtube/invalid_key.json")
    respx_mock.get(SEARCH).mock(return_value=httpx.Response(400, json=body))

    with pytest.raises(ToolError, match="HTTP 400: API key not valid") as error:
        await _tool(client).run(YouTubeInput(query=QUERY))

    assert KEY not in str(error.value)


async def test_a_failure_in_the_later_calls_raises_too(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    _mock(respx_mock, load_fixture)
    body = load_fixture("youtube/quota_exceeded.json")
    respx_mock.get(VIDEOS).mock(return_value=httpx.Response(403, json=body))

    with pytest.raises(ToolError, match="daily quota used up"):
        await _tool(client).run(YouTubeInput(query=QUERY))


@pytest.mark.parametrize("body", ["<html>Bad gateway</html>", json.dumps(["not", "an", "object"]), "{}"])
async def test_server_error_with_an_unusable_body_raises_with_the_status(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, body: str
) -> None:
    respx_mock.get(SEARCH).mock(return_value=httpx.Response(502, text=body))

    with pytest.raises(ToolError, match=r"^youtube: HTTP 502$"):
        await _tool(client).run(YouTubeInput(query=QUERY))


async def test_malformed_json_on_success_raises(client: httpx.AsyncClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(SEARCH).mock(return_value=httpx.Response(200, text="not json"))

    with pytest.raises(ToolError, match="invalid response"):
        await _tool(client).run(YouTubeInput(query=QUERY))


async def test_connection_error_raises_without_the_key(client: httpx.AsyncClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(SEARCH).mock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(ToolError, match="request failed") as error:
        await _tool(client).run(YouTubeInput(query=QUERY))

    assert KEY not in str(error.value)
