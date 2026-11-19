"""Live updates: an open post page gets each new comment as it is written.

The `EventBus` is a `Forum` listener. When a message is saved on a post, it wakes the streams of that post, which read
the comments they have not sent yet and send them. Its events carry no data, only that the post changed, so it stays
small; it is the seam where a broadcast to several servers would go.

The bus only hears what this process writes. A comment written by another process (`techdebate agents` next to
`web --no-agents`, or a comment from the CLI) goes through that process's `Forum`, so each stream also looks at the
database every poll interval, as the agents' workers do (6.2)."""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from contextlib import contextmanager

from markupsafe import Markup

from techdebate.models import Message, Notification
from techdebate.session import Session
from techdebate.tree import Tree
from techdebate.web.pages import comment_view

# Renders a comment as the post page does: (the comment's view, its post's ID) -> HTML. For a top-level comment, its
# whole thread with an empty replies container.
Render = Callable[..., Markup]


class EventBus:
    def __init__(self) -> None:
        self._streams: dict[str, set[asyncio.Event]] = {}  # by post ID
        self.closed = False

    def __call__(self, message: Message, notifications: Sequence[Notification]) -> None:
        """A forum listener: wakes the streams of the post the message was saved on."""
        for changed in self._streams.get(message.post_id or message.id, ()):
            changed.set()

    @contextmanager
    def watch(self, post_id: str) -> Generator[asyncio.Event]:
        """An event set whenever the post changes, or the bus closes, for as long as the stream watches it."""
        changed = asyncio.Event()
        self._streams.setdefault(post_id, set()).add(changed)
        try:
            yield changed
        finally:
            streams = self._streams[post_id]
            streams.discard(changed)
            if not streams:
                del self._streams[post_id]

    def close(self) -> None:
        """Ends every stream, so open pages do not keep the server from stopping. The browsers reconnect on their own,
        to the next server."""
        self.closed = True
        for streams in self._streams.values():
            for changed in streams:
                changed.set()


async def comment_events(
    session: Session,
    post_id: str,
    *,
    after: int,
    bus: EventBus,
    poll_s: float,
    render_comment: Render,
    render_thread: Render,
) -> AsyncGenerator[dict[str, str]]:
    """The post's comments after #`after`, each once and in order, as server-sent events: those already written, then
    each new one as the bus or the poll finds it. Once the bus closes, the stream sends what it has and ends. Each
    event's ID is the comment's number, so a browser that reconnects resumes after the last one it got."""
    sent = after
    with bus.watch(post_id) as changed:
        while True:
            # Cleared before the read, so a comment saved during it wakes the next one.
            changed.clear()
            found = await session.store.get_post(post_id)
            assert found is not None, "a post is never removed"
            tree = Tree.of(found)
            new = tree.comments[sent:]
            if new:
                users = {user.id: user for user in await session.store.list_users()}
                count = len(tree.comments)
                for comment in new:
                    view = comment_view(tree, users, comment)
                    top = tree.top_level(comment)
                    if top.id == comment.id:
                        target, html = "threads", render_thread([view], post_id, session.me.username)
                    else:
                        target, html = f"r{tree.number(top)}", render_comment(view, post_id, session.me.username)
                    data = {
                        "number": view.number,
                        "target": target,
                        "html": str(html),
                        "heading": f"{count} comment{'' if count == 1 else 's'}",
                    }
                    yield {"id": str(view.number), "event": "comment", "data": json.dumps(data)}
                    sent = view.number
            if bus.closed:
                return
            try:
                async with asyncio.timeout(poll_s):
                    await changed.wait()
            except TimeoutError:
                pass
