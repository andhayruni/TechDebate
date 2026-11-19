from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
import respx

from techdebate.config import Settings
from techdebate.errors import NotFoundError, ToolError
from techdebate.tools import PyPITool, Tool, make_http_client
from techdebate.tools.pypi import PyPIInput

PYPI = "pypi.org"
STATS = "pypistats.org"

Fixture = Callable[[str], Any]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with make_http_client(Settings(_env_file=None)) as client:  # pyright: ignore[reportCallIssue]
        yield client


def _mock_httpx(respx_mock: respx.MockRouter, load_fixture: Fixture) -> dict[str, respx.Route]:
    """Routes for every call pypi makes for httpx."""
    return {
        "package": respx_mock.get(host=PYPI, path="/pypi/httpx/json").respond(
            json=load_fixture("pypi/package_httpx.json")
        ),
        "recent": respx_mock.get(host=STATS, path="/api/packages/httpx/recent").respond(
            json=load_fixture("pypi/recent_httpx.json")
        ),
        "overall": respx_mock.get(host=STATS, path="/api/packages/httpx/overall").respond(
            json=load_fixture("pypi/overall_httpx.json")
        ),
    }


def _overall(*days: tuple[str, int]) -> dict[str, Any]:
    rows = [{"category": "without_mirrors", "date": day, "downloads": n} for day, n in days]
    return {"data": rows, "package": "httpx", "type": "overall_downloads"}


def test_tool_satisfies_the_protocol(client: httpx.AsyncClient) -> None:
    tool: Tool[PyPIInput] = PyPITool(client)  # pyright checks the assignment
    assert tool.name == "pypi"


async def test_returns_the_package_recent_and_monthly_downloads(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_httpx(respx_mock, load_fixture)

    info, recent, monthly = await PyPITool(client).run(PyPIInput(query="httpx"))

    assert routes["overall"].calls.last.request.url.params["mirrors"] == "false"
    assert {e.source for e in (info, recent, monthly)} == {"pypi"}
    assert {e.query for e in (info, recent, monthly)} == {"httpx"}

    assert info.title == "httpx"
    assert str(info.url) == "https://pypi.org/project/httpx/"
    assert info.summary == (
        "The next generation HTTP client. "
        "Latest version 0.28.1, released 2024-12-06; first release 2019-07-19; 76 releases."
    )
    assert info.value is None

    assert recent.title == "httpx: recent downloads"
    assert str(recent.url) == "https://pypistats.org/packages/httpx"
    assert recent.summary == (
        "29,604,764 downloads in the last day, 179,482,070 in the last week, "
        "618,500,392 in the last month (without mirrors)"
    )
    assert recent.value == 618_500_392

    assert monthly.title == "httpx: monthly downloads"
    assert str(monthly.url) == "https://pypistats.org/packages/httpx"
    assert "2026-04 (partial, from 04-08): 531,592,651; 2026-05: 728,640,507;" in monthly.summary
    assert "2026-10 (partial, to 10-29): 704,037,655." in monthly.summary
    assert "Total 4,940,599,537 over 205 days (2026-04-08 to 2026-10-29)." in monthly.summary
    assert monthly.value == 4_940_599_537


async def test_monthly_totals_mark_only_partial_months(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_httpx(respx_mock, load_fixture)
    routes["overall"].respond(
        json=_overall(("2026-08-01", 1), ("2026-08-31", 2), ("2026-09-30", 4), ("2026-09-01", 8), ("2026-10-05", 16))
    )

    monthly = (await PyPITool(client).run(PyPIInput(query="httpx")))[2]

    assert monthly.summary == (
        "Monthly downloads without mirrors: 2026-08: 3; 2026-09: 12; 2026-10 (partial, to 10-05): 16. "
        "Total 31 over 66 days (2026-08-01 to 2026-10-05). pypistats.org keeps about 6 months."
    )
    assert monthly.value == 31


async def test_one_partial_month_is_marked_at_both_ends(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_httpx(respx_mock, load_fixture)
    routes["overall"].respond(json=_overall(("2026-10-03", 5), ("2026-10-04", 7)))

    monthly = (await PyPITool(client).run(PyPIInput(query="httpx")))[2]

    assert "2026-10 (partial, from 10-03) (partial, to 10-04): 12." in monthly.summary


async def test_releases_without_files_or_only_yanked_files_are_not_counted(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    package = load_fixture("pypi/package_httpx.json")
    assert package["releases"]["0.0.1"] == []  # recorded as is: a version with no files
    package["releases"]["0.0.2"] = [{"upload_time_iso_8601": "2015-01-02T00:00:00Z", "yanked": True}]
    routes = _mock_httpx(respx_mock, load_fixture)
    routes["package"].respond(json=package)

    info = (await PyPITool(client).run(PyPIInput(query="httpx")))[0]

    # Still 76 releases, but the yanked file is the earliest upload.
    assert info.summary.endswith("first release 2015-01-02; 76 releases.")


async def test_a_mixed_case_name_uses_the_canonical_lowercase_name_for_pypistats(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_httpx(respx_mock, load_fixture)
    respx_mock.get(host=PYPI, path="/pypi/HTTPX/json").respond(json=load_fixture("pypi/package_httpx.json"))

    evidence = await PyPITool(client).run(PyPIInput(query=" HTTPX "))

    assert routes["recent"].called and routes["overall"].called
    assert [e.query for e in evidence] == [" HTTPX "] * 3
    assert evidence[0].title == "httpx"


async def test_unknown_package_raises(client: httpx.AsyncClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host=PYPI, path="/pypi/nope-xx/json").respond(404, json={"message": "Not Found"})

    with pytest.raises(NotFoundError, match=r"^pypi: unknown package nope-xx$"):
        await PyPITool(client).run(PyPIInput(query="nope-xx"))


@pytest.mark.parametrize("query", ["AI in general", "owner/name", "", "-httpx", "httpx."])
async def test_a_query_that_is_not_a_package_name_raises_without_a_request(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, query: str
) -> None:
    with pytest.raises(NotFoundError, match=r"^pypi: not a package name"):
        await PyPITool(client).run(PyPIInput(query=query))

    assert not respx_mock.calls


@pytest.mark.parametrize("route", ["recent", "overall"])
async def test_a_package_without_stats_returns_the_metadata_and_says_so(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture, route: str
) -> None:
    routes = _mock_httpx(respx_mock, load_fixture)
    routes[route].respond(404)

    info, downloads = await PyPITool(client).run(PyPIInput(query="httpx"))

    assert info.title == "httpx"
    assert downloads.title == "httpx: downloads"
    assert str(downloads.url) == "https://pypistats.org/packages/httpx"
    assert downloads.summary == "pypistats.org has no download data for httpx yet"
    assert downloads.value is None


async def test_empty_daily_data_says_so(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_httpx(respx_mock, load_fixture)
    routes["overall"].respond(json=_overall())

    monthly = (await PyPITool(client).run(PyPIInput(query="httpx")))[2]

    assert monthly.summary == "pypistats.org has no daily download data for httpx"
    assert monthly.value is None


async def test_pypistats_rate_limit_raises(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_httpx(respx_mock, load_fixture)
    routes["recent"].respond(429, headers={"Retry-After": "1"})

    with pytest.raises(ToolError, match=r"^pypi: rate limit reached for pypistats\.org$"):
        await PyPITool(client).run(PyPIInput(query="httpx"))


async def test_server_error_raises(client: httpx.AsyncClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(host=PYPI, path="/pypi/httpx/json").respond(500)

    with pytest.raises(ToolError, match=r"^pypi: HTTP 500 for https://pypi\.org/pypi/httpx/json$"):
        await PyPITool(client).run(PyPIInput(query="httpx"))


async def test_connection_error_raises(
    client: httpx.AsyncClient, respx_mock: respx.MockRouter, load_fixture: Fixture
) -> None:
    routes = _mock_httpx(respx_mock, load_fixture)
    routes["overall"].mock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(ToolError, match=r"request failed for https://pypistats\.org/api/packages/httpx/overall: boom$"):
        await PyPITool(client).run(PyPIInput(query="httpx"))
