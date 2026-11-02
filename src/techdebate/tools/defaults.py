"""The registry of every available tool. Each new tool adds one line here."""

from typing import Any

import httpx

from techdebate.config import Settings
from techdebate.tools.arxiv import ArxivTool
from techdebate.tools.base import Tool
from techdebate.tools.github import GitHubRepoTool, GitHubSearchTool
from techdebate.tools.pypi import PyPITool
from techdebate.tools.registry import ToolRegistry
from techdebate.tools.youtube import YouTubeTool


def default_tools(settings: Settings, client: httpx.AsyncClient) -> ToolRegistry:
    """Every available tool, sharing one HTTP client. A tool whose required key is missing is left out."""
    tools: list[Tool[Any]] = [
        GitHubSearchTool(client, settings),  # works without a token, at a lower rate limit
        GitHubRepoTool(client, settings),
        PyPITool(client),
        ArxivTool(client, settings),
    ]
    # An empty YOUTUBE_API_KEY= line in .env counts as no key.
    youtube_key = settings.youtube_api_key.get_secret_value() if settings.youtube_api_key else ""
    if youtube_key:
        tools.append(YouTubeTool(client, youtube_key, settings))
    return ToolRegistry(tools)
