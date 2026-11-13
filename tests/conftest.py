import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import respx

from techdebate.config import Settings
from techdebate.store import Store, open_store

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GITHUB = "api.github.com"
PYPI = "pypi.org"
STATS = "pypistats.org"


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
