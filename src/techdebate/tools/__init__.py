"""Evidence tools: one interface per data source, a registry and a shared HTTP client."""

from techdebate.tools.arxiv import ArxivTool
from techdebate.tools.base import Tool, ToolInput
from techdebate.tools.defaults import default_tools
from techdebate.tools.github import GitHubRepoTool, GitHubSearchTool
from techdebate.tools.http import make_http_client
from techdebate.tools.pypi import PyPITool
from techdebate.tools.registry import ToolRegistry
from techdebate.tools.youtube import YouTubeTool

__all__ = [
    "ArxivTool",
    "GitHubRepoTool",
    "GitHubSearchTool",
    "PyPITool",
    "Tool",
    "ToolInput",
    "ToolRegistry",
    "YouTubeTool",
    "default_tools",
    "make_http_client",
]
