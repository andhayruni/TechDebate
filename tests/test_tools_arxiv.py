import asyncio
import re
from collections.abc import AsyncIterator, Callable
from datetime import date
from typing import Any

import httpx
import pytest
import respx

from techdebate.config import Settings
from techdebate.errors import ToolError
from techdebate.tools import ArxivTool, Tool, make_http_client
from techdebate.tools.arxiv import ArxivInput

HOST = "export.arxiv.org"
PATH = "/api/query"
TODAY = date(2026, 10, 30)
# The count fixture was recorded for 2024; the other years reuse it with their own total.
COUNTS = {2022: 410, 2023: 520, 2024: 370, 2025: 1450, 2026: 980}

TextFixture = Callable[[str], str]


def _settings(**overrides: Any) -> Settings:
    overrides = {"arxiv_request_interval_s": 0, **overrides}
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with make_http_client(_settings()) as client:
        yield client


def _tool(client: httpx.AsyncClient, **settings: Any) -> ArxivTool:
    return ArxivTool(client, _settings(**settings), today=lambda: TODAY)


def _with_total(body: str, total: int) -> str:
    return re.sub(r"<opensearch:totalResults>\d+<", f"<opensearch:totalResults>{total}<", body)


def _responder(
    load: TextFixture, search: str | None = None, seen: list[dict[str, str]] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """Answers every arXiv request: the relevance search, or a count for the year in the query. Each request's query
    parameters are added to `seen`."""
    search_body = search if search is not None else load("arxiv/search_vision_transformer.xml")
    count_body = load("arxiv/count_vision_transformer_2024.xml")

    def respond(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(dict(request.url.params))
        year = re.search(r"submittedDate:\[(\d{4})", request.url.params["search_query"])
        if year is None:
            return httpx.Response(200, text=search_body)
        return httpx.Response(200, text=_with_total(count_body, COUNTS[int(year[1])]))

    return respond


def _mock(respx_mock: respx.MockRouter, load: TextFixture, search: str | None = None) -> list[dict[str, str]]:
    """Routes every arXiv request to `_responder`, and returns the query parameters of each request, in order."""
    seen: list[dict[str, str]] = []
    respx_mock.get(host=HOST, path=PATH).mock(side_effect=_responder(load, search, seen))
    return seen


def test_tool_satisfies_the_protocol(client: httpx.AsyncClient) -> None:
    tool: Tool[ArxivInput] = _tool(client)  # pyright checks the assignment
    assert tool.name == "arxiv"


async def test_returns_the_total_the_trend_and_the_top_papers(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    queries = _mock(respx_mock, load_text_fixture)

    total, trend, *papers = await _tool(client).run(ArxivInput(query="vision transformer"))

    assert queries[0] == {
        "search_query": 'ti:"vision transformer"',
        "sortBy": "relevance",
        "max_results": "5",
    }
    assert {e.source for e in (total, trend, *papers)} == {"arxiv"}
    assert {e.query for e in (total, trend, *papers)} == {"vision transformer"}

    assert total.title == 'arXiv: "vision transformer" in titles'
    assert str(total.url) == "https://arxiv.org/search/?query=%22vision+transformer%22&searchtype=title"
    assert total.summary == '2,017 papers on arXiv have "vision transformer" in the title'
    assert total.value == 2017

    assert trend.title == 'arXiv: "vision transformer" papers per year'
    assert str(trend.url) == (
        "https://arxiv.org/search/?query=%22vision+transformer%22&searchtype=title&order=-announced_date_first"
    )
    assert trend.summary == (
        'Papers with "vision transformer" in the title, by submission year: '
        "2022: 410; 2023: 520; 2024: 370; 2025: 1,450; 2026 (to 10-30): 980."
    )
    assert trend.value == 1450  # the last complete year

    assert [p.title for p in papers] == [
        "Vision Transformer Pruning",
        "Dual Vision Transformer",
        "Peripheral Vision Transformer",
        "Vicinity Vision Transformer",
        "ViTO: Vision Transformer-Operator",
    ]
    first = papers[0]
    assert str(first.url) == "https://arxiv.org/abs/2104.08500"
    assert first.summary.startswith(
        "By Mingjian Zhu, Yehui Tang, Kai Han; submitted 2021-04-17. "
        "Vision transformer has achieved competitive performance on a variety of computer vision applications."
    )
    assert first.value is None
    assert papers[1].summary.startswith("By Ting Yao, Yehao Li, Yingwei Pan et al.; submitted 2022-07-11. ")


async def test_counts_one_request_per_year_up_to_today(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    queries = _mock(respx_mock, load_text_fixture)

    await _tool(client, arxiv_trend_years=3).run(ArxivInput(query="vision transformer"))

    phrase = 'ti:"vision transformer"'
    assert queries[1:] == [
        {"search_query": f"{phrase} AND submittedDate:[202401010000 TO 202412312359]", "max_results": "1"},
        {"search_query": f"{phrase} AND submittedDate:[202501010000 TO 202512312359]", "max_results": "1"},
        {"search_query": f"{phrase} AND submittedDate:[202601010000 TO 202610302359]", "max_results": "1"},
    ]


async def test_a_one_year_trend_has_no_complete_year_for_its_value(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    _mock(respx_mock, load_text_fixture)

    trend = (await _tool(client, arxiv_trend_years=1).run(ArxivInput(query="vision transformer")))[1]

    assert trend.summary.endswith("by submission year: 2026 (to 10-30): 980.")
    assert trend.value is None


async def test_zero_trend_years_skips_the_trend(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    queries = _mock(respx_mock, load_text_fixture)

    evidence = await _tool(client, arxiv_trend_years=0).run(ArxivInput(query="vision transformer"))

    assert len(queries) == 1
    assert [e.title for e in evidence[:2]] == ['arXiv: "vision transformer" in titles', "Vision Transformer Pruning"]
    assert len(evidence) == 6


async def test_no_matches_returns_only_the_zero_count(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    queries = _mock(respx_mock, load_text_fixture, search=load_text_fixture("arxiv/empty.xml"))

    evidence = await _tool(client).run(ArxivInput(query="zzqxvbnmkplw"))

    assert len(queries) == 1  # no yearly counts for a phrase nobody uses
    assert [(e.value, e.summary) for e in evidence] == [(0, '0 papers on arXiv have "zzqxvbnmkplw" in the title')]


async def test_requests_are_spaced_by_the_interval_even_when_runs_overlap(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    now = [100.0]
    sent: list[float] = []
    answer = _responder(load_text_fixture)

    async def sleep(seconds: float) -> None:
        # Like real sleeps, overlapping ones wake together, and the other run gets its turn in the meantime.
        wake = now[0] + seconds
        await asyncio.sleep(0)
        now[0] = max(now[0], wake)

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(now[0])
        return answer(request)

    respx_mock.get(host=HOST, path=PATH).mock(side_effect=respond)
    settings = _settings(arxiv_request_interval_s=3, arxiv_trend_years=2)
    tool = ArxivTool(client, settings, today=lambda: TODAY, sleep=sleep, clock=lambda: now[0])

    await asyncio.gather(*(tool.run(ArxivInput(query="vision transformer")) for _ in range(2)))

    # 2 runs of 3 requests each, one every 3 seconds.
    assert sent == [100.0, 103.0, 106.0, 109.0, 112.0, 115.0]


async def test_time_already_passed_counts_toward_the_interval(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    now = [100.0]
    waits: list[float] = []
    answer = _responder(load_text_fixture)

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    def respond(request: httpx.Request) -> httpx.Response:
        now[0] += 1  # each response takes a second
        return answer(request)

    respx_mock.get(host=HOST, path=PATH).mock(side_effect=respond)
    settings = _settings(arxiv_request_interval_s=3, arxiv_trend_years=1)
    tool = ArxivTool(client, settings, today=lambda: TODAY, sleep=sleep, clock=lambda: now[0])

    await tool.run(ArxivInput(query="vision transformer"))

    assert waits == [2.0]


async def test_the_query_loses_its_quotes_and_extra_whitespace(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    queries = _mock(respx_mock, load_text_fixture)

    evidence = await _tool(client, arxiv_trend_years=0).run(ArxivInput(query='  "vision\n transformer" '))

    assert queries[0]["search_query"] == 'ti:"vision transformer"'
    assert evidence[0].query == '  "vision\n transformer" '
    assert evidence[0].title == 'arXiv: "vision transformer" in titles'


@pytest.mark.parametrize("query", ["", "   ", '""'])
async def test_an_empty_query_raises_without_a_request(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, query: str
) -> None:
    with pytest.raises(ToolError, match=r"^arxiv: empty query$"):
        await _tool(client).run(ArxivInput(query=query))

    assert not respx_mock.calls


async def test_wrapped_titles_and_long_abstracts_are_tidied(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    body = load_text_fixture("arxiv/search_vision_transformer.xml")
    body = body.replace("<title>Vision Transformer Pruning</title>", "<title>Vision Transformer\n  Pruning</title>")
    long_sentence = "word " * 80
    body = body.replace("<summary>Prior works", f"<summary>{long_sentence.strip()}, and more. Prior works", 1)
    _mock(respx_mock, load_text_fixture, search=body)

    papers = (await _tool(client, arxiv_trend_years=0).run(ArxivInput(query="vision transformer")))[1:]

    assert papers[0].title == "Vision Transformer Pruning"
    # Whole sentences up to about 300 characters.
    assert papers[0].summary.endswith("hindering the deployment to mobile devices.")
    # A first sentence longer than that is cut at a word.
    assert papers[1].summary.endswith(" word word…")
    assert len(papers[1].summary.split("2022-07-11. ", 1)[1]) <= 301


async def test_a_paper_without_authors_has_only_its_date(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    body = load_text_fixture("arxiv/search_vision_transformer.xml")
    body = re.sub(r"<author>\s*<name>[^<]*</name>\s*</author>", "", body)
    _mock(respx_mock, load_text_fixture, search=body)

    paper = (await _tool(client, arxiv_trend_years=0).run(ArxivInput(query="vision transformer")))[1]

    assert paper.summary.startswith("Submitted 2021-04-17. Vision transformer has achieved")


async def test_an_error_feed_with_status_500_raises_with_arxivs_message(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    respx_mock.get(host=HOST, path=PATH).respond(500, text=load_text_fixture("arxiv/error.xml"))

    with pytest.raises(ToolError, match=r"^arxiv: HTTP 500: The server encountered an internal error"):
        await _tool(client).run(ArxivInput(query="vision transformer"))


async def test_an_error_feed_with_status_200_raises_with_arxivs_message(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    respx_mock.get(host=HOST, path=PATH).respond(200, text=load_text_fixture("arxiv/error.xml"))

    with pytest.raises(ToolError, match=r"^arxiv: The server encountered an internal error"):
        await _tool(client).run(ArxivInput(query="vision transformer"))


async def test_an_error_in_a_yearly_count_raises(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_text_fixture: TextFixture
) -> None:
    search = load_text_fixture("arxiv/search_vision_transformer.xml")
    respx_mock.get(host=HOST, path=PATH).mock(
        side_effect=[httpx.Response(200, text=search), httpx.Response(500, text=load_text_fixture("arxiv/error.xml"))]
    )

    with pytest.raises(ToolError, match=r"^arxiv: HTTP 500"):
        await _tool(client).run(ArxivInput(query="vision transformer"))


@pytest.mark.parametrize("status", [429, 503])
async def test_rate_limit_or_unavailable_raises(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, status: int
) -> None:
    respx_mock.get(host=HOST, path=PATH).respond(status, text="Rate exceeded.")

    with pytest.raises(ToolError, match=rf"^arxiv: rate limit reached or service unavailable \(HTTP {status}\)$"):
        await _tool(client).run(ArxivInput(query="vision transformer"))


async def test_an_error_status_without_a_feed_raises(client: httpx.AsyncClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host=HOST, path=PATH).respond(400, text="Bad Request")

    with pytest.raises(ToolError, match=r"^arxiv: HTTP 400$"):
        await _tool(client).run(ArxivInput(query="vision transformer"))


@pytest.mark.parametrize("body", ["not xml", "<feed xmlns='http://www.w3.org/2005/Atom'/>"])
async def test_an_invalid_response_raises(client: httpx.AsyncClient, respx_mock: respx.MockRouter, body: str) -> None:
    respx_mock.get(host=HOST, path=PATH).respond(200, text=body)

    with pytest.raises(ToolError, match=r"^arxiv: invalid response"):
        await _tool(client).run(ArxivInput(query="vision transformer"))


async def test_connection_error_raises(client: httpx.AsyncClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host=HOST, path=PATH).mock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(ToolError, match=r"^arxiv: request failed: boom$"):
        await _tool(client).run(ArxivInput(query="vision transformer"))
