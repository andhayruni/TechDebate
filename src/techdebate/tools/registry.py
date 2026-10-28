"""Tools by name, for the analysts, the CLI and (later) the MCP server."""

from collections.abc import Iterable
from typing import Any

from techdebate.errors import ToolError
from techdebate.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any]:
        # The name can come from a user (CLI, MCP), so an unknown one is an expected failure.
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(self._tools) or "none"
            raise ToolError(f"unknown tool: {name} (available: {available})") from None

    def tools(self) -> list[Tool[Any]]:
        """All tools, in the order they were registered."""
        return list(self._tools.values())
