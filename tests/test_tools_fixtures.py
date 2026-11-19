"""The pattern every tool test follows: a recorded response served by respx, and the Evidence the tool makes of it."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from record_fixture import fetch, save

from techdebate.config import Settings
from techdebate.models import Evidence
from techdebate.tools import Tool, ToolInput, make_http_client

ITEMS_URL = "https://api.example.test/items"


class ItemsInput(ToolInput):
    query: str


class ItemsTool:
    """A minimal HTTP-backed tool, standing in for a real one such as GitHub."""

    name = "items"
    description = "Example API: items matching a query, with a score"
    input_model = ItemsInput

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def run(self, input: ItemsInput) -> list[Evidence]:
        response = await self._client.get(ITEMS_URL, params={"q": input.query})
        response.raise_for_status()
        return [
            Evidence(
                source=self.name,
                query=input.query,
                url=item["url"],
                title=item["title"],
                summary=f"score {item['score']}",
                value=item["score"],
            )
            for item in response.json()["items"]
        ]


def _settings() -> Settings:
    return Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


async def test_tool_turns_a_recorded_response_into_evidence(
    respx_mock: respx.MockRouter, load_fixture: Callable[[str], Any]
) -> None:
    route = respx_mock.get(ITEMS_URL, params={"q": "rust"}).respond(json=load_fixture("example/items.json"))

    async with make_http_client(_settings()) as client:
        tool: Tool[ItemsInput] = ItemsTool(client)
        evidence = await tool.run(ItemsInput(query="rust"))

    assert route.called
    assert [(str(e.url), e.title, e.value) for e in evidence] == [
        ("https://api.example.test/items/1", "First item", 42),
        ("https://api.example.test/items/2", "Second item", 7),
    ]
    assert {e.source for e in evidence} == {"items"}
    assert {e.query for e in evidence} == {"rust"}


async def test_recorder_saves_the_body_as_pretty_json(respx_mock: respx.MockRouter, tmp_path: Path) -> None:
    route = respx_mock.get(ITEMS_URL).respond(json={"items": [{"title": "Café"}]})
    out = tmp_path / "example" / "items.json"

    async with httpx.AsyncClient() as client:
        response = await fetch(client, ITEMS_URL, {"Authorization": "Bearer secret"})
    save(response, out)

    assert route.calls.last.request.headers["Authorization"] == "Bearer secret"
    text = out.read_text(encoding="utf-8")
    assert text == json.dumps({"items": [{"title": "Café"}]}, indent=2, ensure_ascii=False) + "\n"
    assert "secret" not in text


async def test_recorder_saves_any_other_body_as_it_is(respx_mock: respx.MockRouter, tmp_path: Path) -> None:
    body = "<?xml version='1.0'?>\n<feed><title>Café</title></feed>\n"
    respx_mock.get(ITEMS_URL).respond(text=body)
    out = tmp_path / "example" / "feed.xml"

    async with httpx.AsyncClient() as client:
        save(await fetch(client, ITEMS_URL), out)

    assert out.read_text(encoding="utf-8") == body


async def test_recorder_fails_on_an_error_status(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(ITEMS_URL).respond(404, json={"message": "Not Found"})

    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.HTTPStatusError):
            await fetch(client, ITEMS_URL)


async def test_recorder_keeps_an_error_body_when_allowed(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(ITEMS_URL).respond(500, text="<feed>error</feed>")

    async with httpx.AsyncClient() as client:
        response = await fetch(client, ITEMS_URL, allow_error=True)

    assert (response.status_code, response.text) == (500, "<feed>error</feed>")
