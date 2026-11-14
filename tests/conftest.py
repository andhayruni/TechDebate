import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from techdebate.config import Settings
from techdebate.store import Store, open_store

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GITHUB = "api.github.com"
PYPI = "pypi.org"
STATS = "pypistats.org"
ARXIV = "export.arxiv.org"
YOUTUBE = "www.googleapis.com"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "techdebate.db"


@pytest.fixture
async def store(db_path: Path) -> AsyncIterator[Store]:
    """A real database in a temporary file, created and migrated as a command would."""
    async with open_store(Settings(_env_file=None, db_path=str(db_path))) as store:  # pyright: ignore[reportCallIssue]
        yield store


@pytest.fixture
def load_fixture() -> Callable[[str], Any]:
    """Load a recorded response body, e.g. load_fixture("github/search_rust.json")."""

    def load(name: str) -> Any:
        return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))

    return load


@pytest.fixture
def load_text_fixture() -> Callable[[str], str]:
    """Load a recorded response body as it is, e.g. load_text_fixture("arxiv/search_vision_transformer.xml")."""

    def load(name: str) -> str:
        return (FIXTURES_DIR / name).read_text(encoding="utf-8")

    return load


@pytest.fixture
def mock_github(respx_mock: respx.MockRouter, load_fixture: Callable[[str], Any]) -> None:
    """Recorded responses for the GitHub tools. github_repo resolves keywords with per_page=1, and every other search
    is github_search's."""
    respx_mock.get(host=GITHUB, path="/search/repositories", params={"per_page": "1"}).respond(
        json=load_fixture("github_repo/search_top_hit.json")
    )
    respx_mock.get(host=GITHUB, path="/search/repositories").respond(json=load_fixture("github_search/rust.json"))
    respx_mock.get(host=GITHUB, path="/repos/fastapi/fastapi").respond(
        json={**load_fixture("github_repo/repo_rust.json"), "full_name": "fastapi/fastapi", "description": "Fast API"}
    )
    respx_mock.get(host=GITHUB, path="/repos/fastapi/fastapi/commits").respond(json=[])
    respx_mock.get(host=GITHUB, path="/repos/fastapi/fastapi/releases/latest").respond(404)
    respx_mock.get(host=GITHUB, path="/search/issues").respond(json={"total_count": 0})


@pytest.fixture
def mock_pypi(respx_mock: respx.MockRouter, load_fixture: Callable[[str], Any]) -> None:
    """Recorded responses for the pypi tool, for the package httpx."""
    respx_mock.get(host=PYPI, path="/pypi/httpx/json").respond(json=load_fixture("pypi/package_httpx.json"))
    respx_mock.get(host=STATS, path="/api/packages/httpx/recent").respond(json=load_fixture("pypi/recent_httpx.json"))
    respx_mock.get(host=STATS, path="/api/packages/httpx/overall").respond(json=load_fixture("pypi/overall_httpx.json"))


@pytest.fixture
def mock_arxiv(respx_mock: respx.MockRouter, load_text_fixture: Callable[[str], str]) -> None:
    """Recorded responses for the arxiv tool: the relevance search, and the same count for every trend year."""
    count = load_text_fixture("arxiv/count_vision_transformer_2024.xml")
    search = load_text_fixture("arxiv/search_vision_transformer.xml")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=count if "submittedDate" in request.url.params["search_query"] else search)

    respx_mock.get(host=ARXIV, path="/api/query").mock(side_effect=respond)


@pytest.fixture
def mock_youtube(respx_mock: respx.MockRouter, load_fixture: Callable[[str], Any]) -> None:
    """Recorded responses for the youtube tool: the search, the search for recent videos, and the statistics."""

    def search(request: httpx.Request) -> httpx.Response:
        name = "recent_rust" if "publishedAfter" in request.url.params else "search_rust"
        return httpx.Response(200, json=load_fixture(f"youtube/{name}.json"))

    respx_mock.get(host=YOUTUBE, path="/youtube/v3/search").mock(side_effect=search)
    respx_mock.get(host=YOUTUBE, path="/youtube/v3/videos").respond(json=load_fixture("youtube/videos_rust.json"))
