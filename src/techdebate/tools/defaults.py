"""The registry of every available tool. Each new tool adds one line here."""

import httpx

from techdebate.config import Settings
from techdebate.tools.arxiv import ArxivTool
from techdebate.tools.github import GitHubRepoTool, GitHubSearchTool
from techdebate.tools.pypi import PyPITool
from techdebate.tools.registry import ToolRegistry


def default_tools(settings: Settings, client: httpx.AsyncClient) -> ToolRegistry:
    """Every available tool, sharing one HTTP client. A tool whose required key is missing is left out."""
    return ToolRegistry(
        [
            GitHubSearchTool(client, settings),  # works without a token, at a lower rate limit
            GitHubRepoTool(client, settings),
            PyPITool(client),
            ArxivTool(client, settings),
        ]
    )
