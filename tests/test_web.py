import asyncio
import re
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jinja2 import UndefinedError

from techdebate.config import Settings
from techdebate.errors import CommandError, ForumError
from techdebate.models import Message
from techdebate.session import Session, open_session
from techdebate.store import open_store
from techdebate.web import app as web_app
from techdebate.web import create_app, get_app_settings, get_session
from techdebate.web.pages import FEED_START_CHARS
from techdebate.web.templating import time_tag

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


def test_the_feed_is_a_page_in_the_base_layout(client: TestClient) -> None:
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


# The feed and the post page


@dataclass(frozen=True)
class _Discussion:
    """A post with two threads. The first has a reply to the top-level comment, and a reply to that reply."""

    post: Message
    top: Message  # #1, by the Developer
    reply: Message  # #2, by me, to #1
    deeper: Message  # #3, by the Scientist, to #2
    second: Message  # #4, by the Scientist, to the post


async def _write(settings: Settings, posts: list[str]) -> list[Message]:
    async with open_session(settings) as session:
        return [await session.forum.create_post(session.me, text) for text in posts]


async def _write_discussion(settings: Settings) -> _Discussion:
    async with open_session(settings) as session:
        forum, me = session.forum, session.me
        developer, scientist = session.agents["developer"], session.agents["scientist"]
        post = await forum.create_post(me, "Is **Rust** growing? See [the survey](https://example.com/survey).")
        top = await forum.create_comment(developer, post.id, "Yes: [rust-lang/rust](https://github.com/rust-lang/rust)")
        reply = await forum.create_comment(me, top.id, "Source for the trend?")
        deeper = await forum.create_comment(scientist, reply.id, "Papers grow too: <script>alert(1)</script>")
        second = await forum.create_comment(scientist, post.id, "A second thread.")
        return _Discussion(post=post, top=top, reply=reply, deeper=deeper, second=second)


def _page(client: TestClient, path: str) -> str:
    response = client.get(path)
    assert response.status_code == 200
    assert _in_layout(response.text)
    return response.text


def test_the_feed_with_no_posts(client: TestClient) -> None:
    assert "No posts yet." in _page(client, "/")


def test_the_feed_lists_the_posts_newest_first_with_their_counts(settings: Settings) -> None:
    older, newer = asyncio.run(_write(settings, ["The older post.", "The newer post."]))
    discussion = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        html = _page(client, "/")

    links = re.findall(r'<a class="feed-item" href="/posts/([^"]+)">', html)
    assert links == [discussion.post.id, newer.id, older.id]
    assert re.findall(r"(\d+) comments?\b", html) == ["4", "0", "0"]


def test_a_feed_item_shows_the_start_as_plain_text(settings: Settings) -> None:
    long_text = "**Bold** start with [a link](https://example.com/hidden). " + "word " * 100
    asyncio.run(_write(settings, [long_text]))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        html = _page(client, "/")

    [start] = re.findall(r'<span class="start">([^<]*)</span>', html)
    assert start.startswith("Bold start with a link. word")
    assert start.endswith("…")
    assert len(start) <= FEED_START_CHARS + 1
    assert "example.com/hidden" not in html


def test_a_feed_item_shows_its_author_and_time(settings: Settings) -> None:
    asyncio.run(_write(settings, ["A post."]))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        html = _page(client, "/")

    assert '<span class="author">me</span> <span class="sep">' in html
    assert "<time datetime=" in html


def test_the_post_page_shows_the_post_and_its_threads_oldest_first(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{d.post.id}")

    assert "<title>Is Rust growing? See the survey. · TechDebate</title>" in html
    assert "<strong>Rust</strong>" in html
    assert "<h2>4 comments</h2>" in html
    positions = [html.index(f'id="c{n}"') for n in (1, 2, 3, 4)]
    assert positions == sorted(positions)
    first, second = html.split('<div class="thread">')[1:]
    # The two replies of the first thread sit together, one level under its top-level comment.
    top, replies = first.split('<div class="replies">')
    assert 'id="c1"' in top
    assert 'id="c2"' in replies
    assert 'id="c3"' in replies
    assert 'id="c4"' in second
    assert '<div class="replies">' not in second


def test_a_reply_to_a_reply_links_to_the_comment_it_answers(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{d.post.id}")

    assert html.count('class="reply-to"') == 1
    deeper = html[html.index('id="c3"') :]
    header = deeper[: deeper.index("</header>")]
    assert 'replying to <a href="#c2">#2</a> by <span class="author">me</span>' in header


def test_the_text_of_a_comment_is_markdown_without_raw_html(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{d.post.id}")

    assert '<a href="https://github.com/rust-lang/rust" target="_blank" rel="nofollow noopener noreferrer">' in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_every_agent_carries_the_ai_badge_and_a_real_user_does_not(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        pages = [_page(client, "/"), _page(client, f"/posts/{d.post.id}")]

    badge = ' <span class="ai" title="AI agent">AI</span>'
    for name in ("Developer", "Scientist"):
        shown = f'<span class="author">{name}</span>'
        assert pages[1].count(shown) >= 1
        assert pages[1].count(shown) == pages[1].count(shown + badge)
    for page in pages:
        assert '<span class="author">me</span>' in page
        assert '<span class="author">me</span>' + badge not in page


def test_a_post_with_no_comments(settings: Settings) -> None:
    [post] = asyncio.run(_write(settings, ["A quiet post."]))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{post.id}")

    assert "<h2>0 comments</h2>" in html
    assert "No comments yet." in html


def test_an_unknown_post_is_the_404_page(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        # A prefix of the ID, a comment's ID and an ID that names nothing.
        for post_id in (d.post.id[:8], d.top.id, "nope"):
            response = client.get(f"/posts/{post_id}")
            assert response.status_code == 404
            assert "There is no post here." in response.text


def test_a_time_is_local_with_the_utc_time_for_machines() -> None:
    moment = datetime(2026, 11, 18, 13, 5, 42, tzinfo=UTC)

    tag = time_tag(moment, ZoneInfo("Europe/Berlin"))

    assert tag == '<time datetime="2026-11-18T13:05:42Z" title="2026-11-18 13:05 UTC">18 Nov 2026, 14:05</time>'
