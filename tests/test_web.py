from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jinja2 import UndefinedError

from techdebate.config import Settings
from techdebate.errors import CommandError, ForumError
from techdebate.session import Session, open_session
from techdebate.store import open_store
from techdebate.web import app as web_app
from techdebate.web import create_app, get_app_settings, get_session

# The app answers only requests addressed to this machine, so the test client uses its address.
BASE_URL = "http://127.0.0.1"


@pytest.fixture
def settings(db_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=str(db_path))  # pyright: ignore[reportCallIssue]


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Runs the app's startup and shutdown around the test."""
    with TestClient(app, base_url=BASE_URL) as client:
        yield client


def _in_layout(html: str) -> bool:
    return '<link rel="stylesheet" href="/static/style.css">' in html and '<a class="brand" href="/">' in html


def test_health_says_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_is_a_page_in_the_base_layout(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert _in_layout(response.text)
    assert "<title>TechDebate</title>" in response.text


def test_the_stylesheet_is_served(client: TestClient) -> None:
    response = client.get("/static/style.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert "prefers-color-scheme: dark" in response.text


def test_no_api_docs(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


# The session and the settings


def test_the_session_is_opened_at_startup_and_closed_at_shutdown(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    @asynccontextmanager
    async def recording(settings: Settings) -> AsyncGenerator[Session]:
        async with open_session(settings) as session:
            events.append("opened")
            yield session
        events.append("closed")

    monkeypatch.setattr(web_app, "open_session", recording)
    app = create_app(settings)

    @app.get("/me")
    async def me(session: Session = Depends(get_session)) -> str:  # pyright: ignore[reportUnusedFunction]  # noqa: B008
        return session.me.username

    with TestClient(app, base_url=BASE_URL) as client:
        assert events == ["opened"]
        assert client.get("/me").json() == "me"
        assert client.get("/me").json() == "me"  # the same session, not one per request
        assert events == ["opened"]
    assert events == ["opened", "closed"]


async def test_startup_creates_the_database_and_syncs_the_users(client: TestClient, settings: Settings) -> None:
    async with open_store(settings) as store:
        usernames = {user.username for user in await store.list_users()}

    assert "me" in usernames
    assert "developer" in usernames


def test_pages_get_the_settings_the_app_was_built_with(settings: Settings) -> None:
    app = create_app(settings)

    @app.get("/db")
    async def db(app_settings: Settings = Depends(get_app_settings)) -> str:  # pyright: ignore[reportUnusedFunction]  # noqa: B008
        return app_settings.db_path

    with TestClient(app, base_url=BASE_URL) as client:
        assert client.get("/db").json() == settings.db_path


def test_without_settings_the_app_uses_the_environment(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_app, "get_settings", lambda: settings)

    assert create_app().state.settings is settings


# Error pages


def test_a_techdebate_error_is_a_400_page_with_its_message(app: FastAPI) -> None:
    @app.get("/fail")
    async def fail() -> None:  # pyright: ignore[reportUnusedFunction]
        raise CommandError("there is no post 3f2a: techdebate posts lists them")

    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/fail")

    assert response.status_code == 400
    assert _in_layout(response.text)
    assert "<h1>400 Bad Request</h1>" in response.text
    assert "<p>there is no post 3f2a: techdebate posts lists them</p>" in response.text


def test_the_message_on_an_error_page_is_escaped(app: FastAPI) -> None:
    @app.get("/fail")
    async def fail() -> None:  # pyright: ignore[reportUnusedFunction]
        raise ForumError("<script>alert(1)</script>")

    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/fail")

    assert "<script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


@pytest.mark.parametrize("path", ["/nope", "/static/nope.css"])
def test_an_unknown_url_is_a_404_page(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert _in_layout(response.text)
    assert "<h1>404 Not Found</h1>" in response.text
    assert "There is no page here." in response.text


def test_a_404_with_a_message_of_its_own_shows_it(app: FastAPI) -> None:
    from starlette.exceptions import HTTPException

    @app.get("/posts/x")
    async def post() -> None:  # pyright: ignore[reportUnusedFunction]
        raise HTTPException(404, "There is no post here.")

    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/posts/x")

    assert response.status_code == 404
    assert "There is no post here." in response.text
    assert "There is no page here." not in response.text


def test_a_wrong_method_keeps_its_status_and_allow_header(client: TestClient) -> None:
    response = client.post("/health")

    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert "<h1>405 Method Not Allowed</h1>" in response.text


def test_an_invalid_value_is_a_422_page(app: FastAPI) -> None:
    @app.get("/number")
    async def number(n: int) -> int:  # pyright: ignore[reportUnusedFunction]
        return n

    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/number", params={"n": "seven"})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("text/html")
    assert "The request has a missing or invalid value." in response.text


def _app_with_a_bug(settings: Settings) -> FastAPI:
    app = create_app(settings)

    @app.get("/bug")
    async def bug() -> None:  # pyright: ignore[reportUnusedFunction]
        raise KeyError("secret /Users/someone/.env")

    return app


def test_an_unexpected_error_is_a_500_page_without_its_details(settings: Settings) -> None:
    with TestClient(_app_with_a_bug(settings), base_url=BASE_URL, raise_server_exceptions=False) as client:
        response = client.get("/bug")

    assert response.status_code == 500
    assert _in_layout(response.text)
    assert "<h1>500 Internal Server Error</h1>" in response.text
    assert "Something went wrong" in response.text
    assert "secret" not in response.text
    assert "KeyError" not in response.text


def test_an_unexpected_error_still_reaches_the_server_which_logs_it(settings: Settings) -> None:
    with TestClient(_app_with_a_bug(settings), base_url=BASE_URL) as client, pytest.raises(KeyError, match="secret"):
        client.get("/bug")


def test_a_missing_template_variable_fails() -> None:
    template = web_app.templates.env.get_template("error.html")

    with pytest.raises(UndefinedError):
        template.render()


# Only this machine


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:8000", "localhost", "localhost:8000"])
def test_requests_to_this_machine_are_answered(client: TestClient, host: str) -> None:
    assert client.get("/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8000", "127.0.0.1.evil.example", "0.0.0.0"])
def test_requests_to_another_host_are_refused(client: TestClient, host: str) -> None:
    """A page that points its own domain at this machine (DNS rebinding) cannot read the app."""
    response = client.get("/", headers={"host": host})

    assert response.status_code == 400
    assert "TechDebate" not in response.text
