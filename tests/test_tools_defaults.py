from typing import Any

import pytest

from techdebate.config import Settings
from techdebate.tools import default_tools, make_http_client


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("token", [None, "secret"])
async def test_default_tools_include_github_pypi_and_arxiv_with_or_without_a_token(token: str | None) -> None:
    settings = _settings(GITHUB_TOKEN=token, YOUTUBE_API_KEY=None)
    async with make_http_client(settings) as client:
        registry = default_tools(settings, client)

    assert [tool.name for tool in registry.tools()] == ["github_search", "github_repo", "pypi", "arxiv"]


@pytest.mark.parametrize("key", [None, ""])
async def test_default_tools_leave_youtube_out_without_a_key(key: str | None) -> None:
    settings = _settings(YOUTUBE_API_KEY=key)
    async with make_http_client(settings) as client:
        registry = default_tools(settings, client)

    assert "youtube" not in [tool.name for tool in registry.tools()]


async def test_default_tools_include_youtube_with_a_key() -> None:
    settings = _settings(YOUTUBE_API_KEY="secret")
    async with make_http_client(settings) as client:
        registry = default_tools(settings, client)

    assert [tool.name for tool in registry.tools()] == ["github_search", "github_repo", "pypi", "arxiv", "youtube"]
