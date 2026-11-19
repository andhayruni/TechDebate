from typing import Any

import httpx
import respx

from techdebate.config import Settings
from techdebate.tools import make_http_client


def _settings(**overrides: Any) -> Settings:
    # Ignore a local .env so the tests see the defaults.
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def test_default_user_agent_names_the_package_and_version() -> None:
    assert _settings().http_user_agent.startswith("techdebate/0.")


async def test_client_uses_timeout_and_user_agent_from_settings() -> None:
    settings = _settings(http_timeout_s=7.5, http_user_agent="techdebate-test/1.0")

    async with make_http_client(settings) as client:
        assert client.timeout == httpx.Timeout(7.5)
        assert client.headers["User-Agent"] == "techdebate-test/1.0"
        assert client.follow_redirects


async def test_requests_carry_the_user_agent() -> None:
    settings = _settings(http_user_agent="techdebate-test/1.0")

    with respx.mock:
        route = respx.get("https://api.example.test/items").respond(json={"ok": True})
        async with make_http_client(settings) as client:
            response = await client.get("https://api.example.test/items")

    assert response.json() == {"ok": True}
    assert route.calls.last.request.headers["User-Agent"] == "techdebate-test/1.0"
