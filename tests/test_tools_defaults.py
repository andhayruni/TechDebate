from typing import Any

import pytest

from techdebate.config import Settings
from techdebate.tools import default_tools, make_http_client


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("token", [None, "secret"])
async def test_default_tools_include_github_pypi_and_arxiv_with_or_without_a_token(token: str | None) -> None:
    settings = _settings(GITHUB_TOKEN=token)
    async with make_http_client(settings) as client:
        registry = default_tools(settings, client)

    assert [tool.name for tool in registry.tools()] == ["github_search", "github_repo", "pypi", "arxiv"]
