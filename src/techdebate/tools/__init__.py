"""Evidence tools: one interface per data source, a registry and a shared HTTP client."""

from techdebate.tools.base import Tool, ToolInput
from techdebate.tools.http import make_http_client
from techdebate.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolInput", "ToolRegistry", "make_http_client"]
