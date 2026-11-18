"""Live updates: the post page gets each new comment, from this process or another, once and in order."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from techdebate.config import Settings
from techdebate.forum import Forum
from techdebate.models import Message
from techdebate.session import Session, open_session
from techdebate.web import create_app
from techdebate.web.events import EventBus, comment_events
from techdebate.web.templating import macro

BASE_URL = "http://127.0.0.1"
# Long enough that an event which arrives in time came from the bus, not from the poll.
NO_POLL = 60.0


@pytest.fixture
def settings(db_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=str(db_path))  # pyright: ignore[reportCallIssue]


@pytest.fixture
async def session(settings: Settings) -> AsyncIterator[Session]:
    async with open_session(settings) as session:
        yield session


@pytest.fixture
def bus(session: Session) -> EventBus:
    bus = EventBus()
    session.forum.subscribe(bus)
    return bus


def _stream(
    session: Session, post: Message, bus: EventBus, *, after: int = 0, poll_s: float = NO_POLL
) -> AsyncGenerator[dict[str, str]]:
    return comment_events(
        session,
        post.id,
        after=after,
        bus=bus,
        poll_s=poll_s,
        render_comment=macro("macros.html", "comment"),
        render_thread=macro("macros.html", "thread"),
    )


async def _next(events: AsyncGenerator[dict[str, str]], timeout_s: float = 2) -> dict[str, Any]:
    event = await asyncio.wait_for(anext(events), timeout_s)
    assert event["event"] == "comment"
    data: dict[str, Any] = json.loads(event["data"])
    assert event["id"] == str(data["number"])
    return data


async def _discussion(session: Session) -> tuple[Message, list[Message]]:
    """A post with #1 by the Developer, #2 by me replying to #1, and #3 by the Scientist on the post."""
    forum, me, agents = session.forum, session.me, session.agents
    post = await forum.create_post(me, "Is Rust growing?")
    first = await forum.create_comment(agents["developer"], post.id, "Yes.")
    reply = await forum.create_comment(me, first.id, "Source?")
    second = await forum.create_comment(agents["scientist"], post.id, "Papers too.")
    return post, [first, reply, second]


async def test_the_stream_sends_the_comments_after_the_ones_the_page_shows(session: Session, bus: EventBus) -> None:
    post, _ = await _discussion(session)
    events = _stream(session, post, bus, after=1)

    reply, second = await _next(events), await _next(events)
    await events.aclose()

    assert (reply["number"], reply["target"], reply["heading"]) == (2, "r1", "3 comments")
    assert (second["number"], second["target"]) == (3, "threads")


async def test_a_reply_goes_to_its_thread_and_a_top_level_comment_is_a_new_thread(
    session: Session, bus: EventBus
) -> None:
    post, _ = await _discussion(session)
    events = _stream(session, post, bus, after=1)

    reply, second = await _next(events), await _next(events)
    await events.aclose()

    assert reply["html"].startswith('<article class="comment" id="c2">')
    assert second["html"].startswith('<div class="thread" id="t3">')
    assert '<div class="replies" id="r3"></div>' in second["html"]


async def test_a_comment_from_this_process_arrives_at_once(session: Session, bus: EventBus) -> None:
    post = await session.forum.create_post(session.me, "Is Rust growing?")
    events = _stream(session, post, bus)
    waiting = asyncio.ensure_future(_next(events))
    await asyncio.sleep(0.05)  # the stream is waiting

    await session.forum.create_comment(session.agents["developer"], post.id, "Yes.")

    first = await waiting  # well within NO_POLL: the bus woke it
    await events.aclose()
    assert (first["number"], first["heading"]) == (1, "1 comment")


async def test_a_comment_from_another_process_arrives_with_the_poll(session: Session, bus: EventBus) -> None:
    post = await session.forum.create_post(session.me, "Is Rust growing?")
    events = _stream(session, post, bus, poll_s=0.05)
    waiting = asyncio.ensure_future(_next(events))
    await asyncio.sleep(0.05)

    # Another process has a forum of its own, which this bus never hears.
    elsewhere = Forum(session.store, session.agents.values())
    await elsewhere.create_comment(session.agents["developer"], post.id, "Yes.")

    assert (await waiting)["number"] == 1
    await events.aclose()


async def test_each_comment_is_sent_once_and_in_order(session: Session, bus: EventBus) -> None:
    post = await session.forum.create_post(session.me, "Is Rust growing?")
    events = _stream(session, post, bus, poll_s=0.01)
    for n in range(4):
        await session.forum.create_comment(session.agents["developer"], post.id, f"Comment {n}.")

    numbers = [(await _next(events))["number"] for _ in range(4)]
    with pytest.raises(TimeoutError):  # and nothing more, though the poll and the bus both ran meanwhile
        await _next(events, timeout_s=0.2)
    await events.aclose()

    assert numbers == [1, 2, 3, 4]


async def test_closing_the_bus_ends_the_stream(session: Session, bus: EventBus) -> None:
    post = await session.forum.create_post(session.me, "Is Rust growing?")
    events = _stream(session, post, bus)
    waiting = asyncio.ensure_future(anext(events))
    await asyncio.sleep(0.05)

    bus.close()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(waiting, 2)


async def test_the_bus_wakes_only_the_streams_of_the_post_written_on(session: Session, bus: EventBus) -> None:
    quiet = await session.forum.create_post(session.me, "A quiet post.")
    busy = await session.forum.create_post(session.me, "A busy post.")

    with bus.watch(quiet.id) as quiet_changed, bus.watch(busy.id) as busy_changed:
        await session.forum.create_comment(session.agents["developer"], busy.id, "Yes.")

        assert busy_changed.is_set()
        assert not quiet_changed.is_set()


# The endpoint and the page


def _events_of(body: str) -> list[dict[str, Any]]:
    """The comment events in a whole stream."""
    found: list[dict[str, Any]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if fields.get("event") == "comment":
            found.append({"id": fields["id"], **json.loads(fields["data"])})
    return found


def _closed_stream(settings: Settings, path: str, **headers: str) -> tuple[int, str, str]:
    """The status, type and body of a stream from an app whose bus is closed, so the stream sends what it has and
    ends."""
    app = create_app(settings, agents=False)
    with TestClient(app, base_url=BASE_URL) as client:
        app.state.events.close()
        response = client.get(path, headers=headers)
        return response.status_code, response.headers["content-type"], response.text


def test_the_endpoint_streams_the_comments_after_the_page(settings: Settings) -> None:
    post, _ = asyncio.run(_write_discussion(settings))

    status, content_type, body = _closed_stream(settings, f"/posts/{post.id}/events?after=1")

    assert status == 200
    assert content_type.startswith("text/event-stream")
    assert [(e["id"], e["target"]) for e in _events_of(body)] == [("2", "r1"), ("3", "threads")]


def test_a_reconnecting_browser_resumes_after_the_last_comment_it_got(settings: Settings) -> None:
    post, _ = asyncio.run(_write_discussion(settings))

    _, _, body = _closed_stream(settings, f"/posts/{post.id}/events?after=0", **{"last-event-id": "2"})

    assert [e["id"] for e in _events_of(body)] == ["3"]


def test_a_live_comment_looks_as_it_does_after_a_reload(settings: Settings) -> None:
    post, _ = asyncio.run(_write_discussion(settings))

    _, _, body = _closed_stream(settings, f"/posts/{post.id}/events")
    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        page = client.get(f"/posts/{post.id}").text

    events = {event["number"]: event["html"] for event in _events_of(body)}
    for html in events.values():  # each comment, as sent, is on the reloaded page
        article = html[html.index("<article") : html.index("</article>") + len("</article>")]
        assert article in page
    # A thread is sent with its replies container empty, which is how the page shows a thread with no replies (#3).
    assert events[3] in page


def test_the_events_of_an_unknown_post_are_the_404_page(settings: Settings) -> None:
    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        response = client.get("/posts/nope/events")

    assert response.status_code == 404
    assert "There is no post here." in response.text


def test_the_post_page_starts_its_stream_after_the_comments_it_shows(settings: Settings) -> None:
    post, _ = asyncio.run(_write_discussion(settings))

    with TestClient(create_app(settings, agents=False), base_url=BASE_URL) as client:
        page = client.get(f"/posts/{post.id}").text
        script = client.get("/static/live.js")

    assert f'data-events="/posts/{post.id}/events?after=3"' in page
    assert '<script src="/static/live.js" defer></script>' in page
    assert script.status_code == 200
    assert "new EventSource(" in script.text


async def _write_discussion(settings: Settings) -> tuple[Message, list[Message]]:
    async with open_session(settings) as session:
        return await _discussion(session)
