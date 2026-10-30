"""Evidence tools: one interface per data source, a registry and a shared HTTP client."""

from techdebate.tools.base import Tool, ToolInput
from techdebate.tools.defaults import default_tools
from techdebate.tools.github import GitHubRepoTool, GitHubSearchTool
from techdebate.tools.http import make_http_client
from techdebate.tools.registry import ToolRegistry

__all__ = [
    "GitHubRepoTool",
    "GitHubSearchTool",
    "Tool",
    "ToolInput",
    "ToolRegistry",
    "default_tools",
    "make_http_client",
]
