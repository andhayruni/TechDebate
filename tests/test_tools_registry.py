import pytest
from pydantic import Field, HttpUrl, ValidationError

from techdebate.errors import ToolError
from techdebate.models import Evidence
from techdebate.tools import Tool, ToolInput, ToolRegistry


class StubInput(ToolInput):
    query: str = Field(description="What to search for")


class StubTool:
    input_model = StubInput

    def __init__(self, name: str = "stub") -> None:
        self.name = name
        self.description = f"{name}: finds stub facts"

    async def run(self, input: StubInput) -> list[Evidence]:
        url = HttpUrl("https://example.test/")
        return [Evidence(source=self.name, query=input.query, url=url, title="t", summary="s")]


def test_stub_tool_satisfies_the_protocol() -> None:
    tool: Tool[StubInput] = StubTool()  # pyright checks the assignment
    assert tool.input_model is StubInput


def test_registry_lists_tools_in_registration_order() -> None:
    a, b, c = StubTool("b"), StubTool("a"), StubTool("c")
    registry = ToolRegistry([a, b])
    registry.register(c)

    assert registry.tools() == [a, b, c]


def test_registry_looks_up_by_name() -> None:
    tool = StubTool("github")
    assert ToolRegistry([tool]).get("github") is tool


def test_duplicate_name_is_rejected() -> None:
    registry = ToolRegistry([StubTool("github")])
    with pytest.raises(ValueError, match="already registered: github"):
        registry.register(StubTool("github"))


def test_unknown_name_raises_tool_error_listing_the_available_tools() -> None:
    registry = ToolRegistry([StubTool("github"), StubTool("pypi")])
    with pytest.raises(ToolError, match=r"unknown tool: arxiv \(available: github, pypi\)"):
        registry.get("arxiv")


def test_unknown_name_in_an_empty_registry() -> None:
    with pytest.raises(ToolError, match=r"\(available: none\)"):
        ToolRegistry().get("github")


async def test_untrusted_input_is_validated_through_input_model() -> None:
    tool = ToolRegistry([StubTool()]).get("stub")

    evidence = await tool.run(tool.input_model.model_validate({"query": "rust"}))

    assert [e.query for e in evidence] == ["rust"]


def test_input_schema_rejects_unknown_fields() -> None:
    schema = StubInput.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["query"]["description"] == "What to search for"

    with pytest.raises(ValidationError):
        StubInput.model_validate({"query": "rust", "limit": 5})
