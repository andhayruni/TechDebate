import asyncio
import re
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from httpx2 import Response
from jinja2 import UndefinedError

from techdebate.config import Settings
from techdebate.errors import CommandError, ForumError
from techdebate.llm import FakeLLM, LLMRequest
from techdebate.llm.fake import Predicate, Reply
from techdebate.models import LLMCall, Message, ReplyTo, ResearchPlan, SpeakDecision
from techdebate.orchestrator import Visit
from techdebate.roles import ROLE_SPECS
from techdebate.runner import agents_lock
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
    return create_app(settings, agents=False)


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
    app = create_app(settings, agents=False)

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
    app = create_app(settings, agents=False)

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
    app = create_app(settings, agents=False)

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

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, "/")

    links = re.findall(r'<a class="feed-item" href="/posts/([^"]+)">', html)
    assert links == [discussion.post.id, newer.id, older.id]
    assert re.findall(r"(\d+) comments?\b", html) == ["4", "0", "0"]


def test_a_feed_item_shows_the_start_as_plain_text(settings: Settings) -> None:
    long_text = "**Bold** start with [a link](https://example.com/hidden). " + "word " * 100
    asyncio.run(_write(settings, [long_text]))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, "/")

    [start] = re.findall(r'<span class="start">([^<]*)</span>', html)
    assert start.startswith("Bold start with a link. word")
    assert start.endswith("…")
    assert len(start) <= FEED_START_CHARS + 1
    assert "example.com/hidden" not in html


def test_a_feed_item_shows_its_author_and_time(settings: Settings) -> None:
    asyncio.run(_write(settings, ["A post."]))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, "/")

    assert '<span class="author">me</span> <span class="sep">' in html
    assert "<time datetime=" in html


def test_the_post_page_shows_the_post_and_its_threads_oldest_first(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{d.post.id}")

    assert "<title>Is Rust growing? See the survey. · TechDebate</title>" in html
    assert "<strong>Rust</strong>" in html
    assert '<h2 id="comment-count">4 comments</h2>' in html
    positions = [html.index(f'id="c{n}"') for n in (1, 2, 3, 4)]
    assert positions == sorted(positions)
    first, second = html.split('<div class="thread"')[1:]
    # The two replies of the first thread sit together, one level under its top-level comment.
    top, replies = first.split('<div class="replies" id="r1">')
    assert 'id="c1"' in top
    assert 'id="c2"' in replies
    assert 'id="c3"' in replies
    # A thread with no replies has an empty container, for a reply that arrives live.
    assert 'id="c4"' in second
    assert '<div class="replies" id="r4"></div>' in second


def test_a_reply_to_a_reply_links_to_the_comment_it_answers(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{d.post.id}")

    assert html.count('class="reply-to"') == 1
    deeper = html[html.index('id="c3"') :]
    header = deeper[: deeper.index("</header>")]
    assert 'replying to <a href="#c2">#2</a> by <span class="author">me</span>' in header


def test_the_text_of_a_comment_is_markdown_without_raw_html(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{d.post.id}")

    assert '<a href="https://github.com/rust-lang/rust" target="_blank" rel="nofollow noopener noreferrer">' in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_every_agent_carries_the_ai_badge_and_a_real_user_does_not(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
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

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{post.id}")

    assert '<h2 id="comment-count">0 comments</h2>' in html
    assert "No comments yet." in html


def test_an_unknown_post_is_the_404_page(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        # A prefix of the ID, a comment's ID and an ID that names nothing.
        for post_id in (d.post.id[:8], d.top.id, "nope"):
            response = client.get(f"/posts/{post_id}")
            assert response.status_code == 404
            assert "There is no post here." in response.text


def test_a_time_is_local_with_the_utc_time_for_machines() -> None:
    moment = datetime(2026, 11, 18, 13, 5, 42, tzinfo=UTC)

    tag = time_tag(moment, ZoneInfo("Europe/Berlin"))

    assert tag == '<time datetime="2026-11-18T13:05:42Z" title="2026-11-18 13:05 UTC">18 Nov 2026, 14:05</time>'


# Writing from the browser

OWN_ORIGIN = {"origin": BASE_URL}


async def _read(settings: Settings) -> tuple[list[Message], dict[str, list[tuple[str, str]]]]:
    """Every message, oldest first, and each user's notifications as (reason, message ID), by username."""
    async with open_store(settings) as store:
        messages: list[Message] = []
        for post in reversed(await store.list_posts()):
            found = await store.get_post(post.id)
            assert found is not None
            messages += [found.post, *found.comments]
        notified: dict[str, list[tuple[str, str]]] = {}
        for user in await store.list_users():
            notifications = await store.list_notifications(user.id, include_closed=True)
            notified[user.username] = [(n.reason.value, n.message_id) for n in notifications]
        return messages, notified


def _write_from_page(client: TestClient, path: str, data: dict[str, str], **headers: str) -> Response:
    return client.post(path, data=data, headers=headers or OWN_ORIGIN, follow_redirects=False)


def test_a_post_written_from_the_feed_lands_on_its_own_page(client: TestClient, settings: Settings) -> None:
    response = _write_from_page(client, "/posts", {"text": "  Is **Rust** growing?  "})

    messages, notified = asyncio.run(_read(settings))
    [post] = messages
    assert response.status_code == 303
    assert response.headers["location"] == f"/posts/{post.id}"
    assert post.text == "Is **Rust** growing?"  # trimmed by Forum, as from the CLI
    assert notified["developer"] == [("new_post", post.id)]
    assert notified["scientist"] == [("new_post", post.id)]
    assert "<strong>Rust</strong>" in client.get(response.headers["location"]).text


def test_a_comment_lands_back_on_its_post_at_the_comment(client: TestClient, settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    response = _write_from_page(client, f"/posts/{d.post.id}/comments", {"text": "A new thread."})

    messages, _ = asyncio.run(_read(settings))
    comment = messages[-1]
    assert response.status_code == 303
    assert response.headers["location"] == f"/posts/{d.post.id}#c5"
    assert comment.parent_id == d.post.id
    assert comment.text == "A new thread."


def test_a_reply_lands_under_the_comment_it_answers(client: TestClient, settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    response = _write_from_page(client, f"/posts/{d.post.id}/comments", {"text": "Which papers?", "reply_to": "3"})

    messages, notified = asyncio.run(_read(settings))
    reply = messages[-1]
    assert response.headers["location"] == f"/posts/{d.post.id}#c5"
    assert reply.parent_id == d.deeper.id
    # As from the CLI: the Scientist, who wrote in the thread, is notified.
    assert ("thread", reply.id) in notified["scientist"]
    html = client.get(f"/posts/{d.post.id}").text
    header = html[html.index('id="c5"') :]
    assert 'replying to <a href="#c3">#3</a>' in header[: header.index("</header>")]


def test_a_mention_notifies_the_agent_named(client: TestClient, settings: Settings) -> None:
    [post] = asyncio.run(_write(settings, ["Is Rust growing?"]))

    _write_from_page(client, f"/posts/{post.id}/comments", {"text": "@Scientist what do the papers say?"})

    messages, notified = asyncio.run(_read(settings))
    assert ("mention", messages[-1].id) in notified["scientist"]
    assert ("mention", messages[-1].id) not in notified["developer"]


@pytest.mark.parametrize("path", ["/posts", "comments"])
def test_a_blank_text_is_refused_with_the_400_page(client: TestClient, settings: Settings, path: str) -> None:
    [post] = asyncio.run(_write(settings, ["A post."]))
    path = f"/posts/{post.id}/comments" if path == "comments" else path

    response = _write_from_page(client, path, {"text": "  \n  "})

    assert response.status_code == 400
    assert "the text is blank" in response.text
    messages, _ = asyncio.run(_read(settings))
    assert messages == [post]


def test_a_comment_on_an_unknown_post_or_number_is_the_404_page(client: TestClient, settings: Settings) -> None:
    [post] = asyncio.run(_write(settings, ["A post."]))

    unknown_post = _write_from_page(client, "/posts/nope/comments", {"text": "Hello"})
    unknown_number = _write_from_page(client, f"/posts/{post.id}/comments", {"text": "Hello", "reply_to": "4"})

    assert unknown_post.status_code == 404
    assert "There is no post here." in unknown_post.text
    assert unknown_number.status_code == 404
    assert "There is no comment #4 on this post." in unknown_number.text
    messages, _ = asyncio.run(_read(settings))
    assert messages == [post]


def test_a_reply_cannot_name_a_comment_by_anything_but_its_number(client: TestClient, settings: Settings) -> None:
    [post] = asyncio.run(_write(settings, ["A post."]))

    response = _write_from_page(client, f"/posts/{post.id}/comments", {"text": "Hello", "reply_to": "abc"})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "http://127.0.0.1"},
        {"origin": "HTTP://127.0.0.1"},
        {"sec-fetch-site": "same-origin"},
        {"origin": "http://127.0.0.1", "sec-fetch-site": "same-origin"},
    ],
)
def test_a_write_from_the_apps_own_pages_is_accepted(
    client: TestClient, settings: Settings, headers: dict[str, str]
) -> None:
    response = _write_from_page(client, "/posts", {"text": "Hello"}, **headers)

    assert response.status_code == 303


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"origin": "https://evil.example"},
        {"origin": "null"},
        {"origin": "http://localhost"},  # another origin than the one the request was sent to
        {"origin": "http://127.0.0.1:9999"},
        {"origin": "https://127.0.0.1"},
        {"sec-fetch-site": "cross-site"},
        {"sec-fetch-site": "same-site"},
        {"sec-fetch-site": "none"},
    ],
)
def test_a_write_from_anywhere_else_is_refused_with_the_403_page(
    client: TestClient, settings: Settings, headers: dict[str, str]
) -> None:
    response = client.post("/posts", data={"text": "Hello"}, headers=headers, follow_redirects=False)

    assert response.status_code == 403
    assert _in_layout(response.text)
    assert "Only the app&#39;s own pages can write." in response.text
    messages, _ = asyncio.run(_read(settings))
    assert messages == []


def test_the_origin_check_comes_before_the_form_is_checked(client: TestClient) -> None:
    """A refused request learns nothing, not even that its form was incomplete."""
    assert client.post("/posts", data={}, headers={"origin": "https://evil.example"}).status_code == 403


def test_reading_needs_no_origin(client: TestClient) -> None:
    assert client.get("/", headers={"origin": "https://evil.example"}).status_code == 200


def test_the_feed_has_the_box_for_a_post_naming_the_user(db_path: Path) -> None:
    settings = Settings(_env_file=None, db_path=str(db_path), username="andok")  # pyright: ignore[reportCallIssue]

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, "/")

    assert '<form class="write" method="post" action="/posts">' in html
    assert '<textarea name="text" rows="3" required' in html
    assert "Writing as andok" in html
    assert "No posts yet. Write the first one above." in html


def test_the_post_page_has_a_box_for_a_comment_and_one_for_a_reply_under_each(settings: Settings) -> None:
    d = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        html = _page(client, f"/posts/{d.post.id}")

    action = f'action="/posts/{d.post.id}/comments"'
    assert html.count(action) == 5  # the comment box and one reply box per comment
    for n in (1, 2, 3, 4):
        comment = html[html.index(f'id="c{n}"') :]
        comment = comment[: comment.index("</article>")]
        assert '<details class="reply">' in comment
        assert f'<input type="hidden" name="reply_to" value="{n}">' in comment
    top_box = html[html.index("</article>") : html.index('<section class="comments"')]
    assert action in top_box
    assert "reply_to" not in top_box


# The agents in the app


def _developer_answers() -> FakeLLM:
    """The Developer researches with no queries, so no tool makes a request, and comments on the post. Every other
    agent passes at research."""

    def research_by(display_name: str) -> Predicate:
        def matches(request: LLMRequest) -> bool:
            return request.prompt_id == "research.v1" and request.system.startswith(f"You are the {display_name},")

        return matches

    def prompt(prompt_id: str) -> Predicate:
        def matches(request: LLMRequest) -> bool:
            return request.prompt_id == prompt_id

        return matches

    rules: list[tuple[Predicate, Reply]] = [
        (
            research_by(role.display_name),
            ResearchPlan(
                speak=role.name == "developer", queries={name: [] for name in role.tool_names}, reason="Checked."
            ),
        )
        for role in ROLE_SPECS
    ]
    rules += [
        (prompt("decide.v2"), SpeakDecision(reason="New data.", reply_to=[ReplyTo(place=0, point="It is used.")])),
        (prompt("developer.v1"), "Rust is used, by the numbers."),
    ]
    return FakeLLM(rules=rules)


def _wait_for(check: Callable[[], bool], timeout_s: float = 10) -> None:
    deadline = time.monotonic() + timeout_s
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def test_a_post_written_in_the_browser_is_answered_by_the_agents_in_the_app(settings: Settings) -> None:
    visits: list[Visit] = []
    app = create_app(settings, llm=_developer_answers(), on_visit=visits.append)

    with TestClient(app, base_url=BASE_URL) as client:
        location = _write_from_page(client, "/posts", {"text": "Is Rust growing?"}).headers["location"]
        _wait_for(lambda: "Rust is used, by the numbers." in client.get(location).text)
        html = client.get(location).text

    assert '<span class="author">Developer</span> <span class="ai" title="AI agent">AI</span>' in html
    assert '<h2 id="comment-count">1 comment</h2>' in html
    assert any(visit.agent.username == "developer" and visit.comments == (1,) for visit in visits)


def test_the_app_holds_the_agents_lock_while_it_runs_and_releases_it(settings: Settings) -> None:
    app = create_app(settings, llm=_developer_answers())

    with TestClient(app, base_url=BASE_URL):
        assert app.state.runner is not None
        with pytest.raises(CommandError, match="already running"), agents_lock(settings):
            pass

    assert app.state.runner is None
    with agents_lock(settings):  # released on shutdown
        pass


def test_the_app_does_not_start_while_another_process_runs_the_agents(settings: Settings) -> None:
    with agents_lock(settings), pytest.raises(CommandError, match="already running"):
        with TestClient(create_app(settings, llm=_developer_answers()), base_url=BASE_URL):
            pass


def test_without_agents_the_app_takes_no_lock(settings: Settings) -> None:
    app = create_app(settings, agents=False)

    with agents_lock(settings), TestClient(app, base_url=BASE_URL) as client:
        assert client.get("/").status_code == 200
        assert app.state.runner is None


# The budgets on the pages


async def _spend(settings: Settings, post: Message, *costs: tuple[str, float]) -> None:
    """Stores an AI call per (agent, cost) on the post, made now."""
    async with open_store(settings) as store:
        for username, cost in costs:
            agent = await store.find_user(username)
            assert agent is not None
            await store.add_llm_call(
                LLMCall(
                    agent_id=agent.id,
                    post_id=post.id,
                    prompt_id="decide.v2",
                    model="claude-haiku-5-5",
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=cost,
                    duration_s=0.1,
                )
            )


def _budget_settings(db_path: Path, **limits: float) -> Settings:
    return Settings(_env_file=None, db_path=str(db_path), **limits)  # pyright: ignore[reportCallIssue]


def _pages(settings: Settings, post: Message) -> list[str]:
    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        return [_page(client, path) for path in ("/", f"/posts/{post.id}", "/budget")]


def _text(html: str) -> str:
    """What a reader sees of some HTML: its text, with the tags taken out and the spaces collapsed."""
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def _notices_in(html: str) -> list[str]:
    return [_text(notice) for notice in re.findall(r'<div class="notice"[^>]*>(.*?)</div>', html, re.S)]


def test_no_notice_while_no_budget_is_used_up(settings: Settings) -> None:
    [post] = asyncio.run(_write(settings, ["A post."]))
    asyncio.run(_spend(settings, post, ("developer", 0.01)))

    for html in _pages(settings, post):
        assert _notices_in(html) == []


def test_the_day_used_up_shows_on_every_page(db_path: Path) -> None:
    settings = _budget_settings(db_path, daily_budget_usd=0.05)
    [post] = asyncio.run(_write(settings, ["A post."]))
    asyncio.run(_spend(settings, post, ("developer", 0.03), ("scientist", 0.03)))

    for html in _pages(settings, post):
        [notice] = _notices_in(html)
        assert notice.startswith(
            "The agents are out of budget for today ($0.05). They'll answer again after the reset in "
        )
        assert notice.endswith("Reading and writing still work. Budget")


def test_an_agent_used_up_shows_on_every_page_with_its_badge(db_path: Path) -> None:
    settings = _budget_settings(db_path, agent_budget_usd=0.02)
    [post] = asyncio.run(_write(settings, ["A post."]))
    asyncio.run(_spend(settings, post, ("developer", 0.03), ("scientist", 0.01)))

    for html in _pages(settings, post):
        [notice] = _notices_in(html)
        assert notice.startswith(
            "Developer AI is out of its budget for today ($0.02). It'll answer again after the reset in "
        )
        assert '<span class="ai" title="AI agent">AI</span> is out of its budget' in html


def test_a_post_used_up_shows_on_its_page_only(db_path: Path) -> None:
    settings = _budget_settings(db_path, post_budget_usd=0.02)
    spent, other = asyncio.run(_write(settings, ["Spent post.", "Other post."]))
    asyncio.run(_spend(settings, spent, ("developer", 0.03)))

    feed_page, post_page, budget_page = _pages(settings, spent)
    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        other_page = _page(client, f"/posts/{other.id}")

    assert _notices_in(post_page) == [
        "The agents are out of budget for this post ($0.02) and won't comment on it again. Budget"
    ]
    for html in (feed_page, budget_page, other_page):
        assert _notices_in(html) == []


def test_the_budget_page_shows_today_and_every_post(db_path: Path) -> None:
    settings = _budget_settings(db_path, post_budget_usd=0.02)
    older, _ = asyncio.run(_write(settings, ["The older post.", "The newer post."]))
    asyncio.run(_spend(settings, older, ("developer", 0.03), ("scientist", 0.0012)))

    html = _pages(settings, older)[2]

    rows = [_text(row) for row in re.findall(r"<tr>(.*?)</tr>", html, re.S)]
    assert rows == [
        "All agents $0.0312 of $5.00 $4.9688 left",
        "Developer AI $0.0300 of $2.00 $1.9700 left",
        "Scientist AI $0.0012 of $2.00 $1.9988 left",
        "The newer post. $0.0000 of $0.02 $0.0200 left",
        "The older post. $0.0312 of $0.02 used up",
    ]
    assert re.search(r"\(UTC, resets in \d+h \d+m\)", html)
    assert f'<a href="/posts/{older.id}">The older post.</a>' in html
    assert '<nav><a href="/budget">Budget</a></nav>' in html
