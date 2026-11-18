"""The pages: the feed, and a post with its threads, and the forms that write to them. Reads go through the `Store`,
and writes through `Forum` as the user in `username`, so they are checked and notified as on the command line."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException

from techdebate.session import Session
from techdebate.store import PostWithComments, Store
from techdebate.tree import Tree
from techdebate.web.deps import get_session
from techdebate.web.pages import feed, post_page
from techdebate.web.templating import templates

router = APIRouter(default_response_class=HTMLResponse)

CurrentSession = Annotated[Session, Depends(get_session)]

_NO_POST = "There is no post here."


@router.get("/", name="home")
async def home(request: Request, session: CurrentSession) -> Response:
    """The feed: the posts, newest first, and the box for a new one."""
    store = session.store
    users = {user.id: user for user in await store.list_users()}
    items = feed(await store.list_post_summaries(), users)
    return templates.TemplateResponse(request, "feed.html", {"posts": items, "me": session.me.username})


@router.get("/posts/{post_id}", name="post")
async def show_post(request: Request, post_id: str, session: CurrentSession) -> Response:
    """A post and its threads, with the boxes for a comment and the replies. The ID is the post's full ID."""
    store = session.store
    found = await _post(store, post_id)
    users = {user.id: user for user in await store.list_users()}
    context = {"page": post_page(Tree.of(found), users), "me": session.me.username}
    return templates.TemplateResponse(request, "post.html", context)


@router.post("/posts")
async def write_post(text: Annotated[str, Form()], session: CurrentSession) -> Response:
    """A new post, then its own page, where its discussion happens."""
    post = await session.forum.create_post(session.me, text)
    return _see(f"/posts/{post.id}")


@router.post("/posts/{post_id}/comments")
async def write_comment(
    post_id: str,
    text: Annotated[str, Form()],
    session: CurrentSession,
    reply_to: Annotated[int | None, Form()] = None,
) -> Response:
    """A top-level comment, or a reply to comment #reply_to, then the post at the new comment. The number is looked up
    on this post, so a reply cannot land on another post."""
    store = session.store
    tree = Tree.of(await _post(store, post_id))
    parent = tree.post if reply_to is None else tree.comment(reply_to)
    if parent is None:
        raise HTTPException(404, f"There is no comment #{reply_to} on this post.")
    comment = await session.forum.create_comment(session.me, parent.id, text)
    # Read again for the number: an agent may have commented since the tree above was read.
    number = Tree.of(await _post(store, post_id)).number(comment)
    return _see(f"/posts/{post_id}#c{number}")


async def _post(store: Store, post_id: str) -> PostWithComments:
    found = await store.get_post(post_id)
    if found is None:
        raise HTTPException(404, _NO_POST)
    return found


def _see(url: str) -> Response:
    """After a write: 303, so the browser loads the page with GET, and reloading it never sends the form again."""
    return RedirectResponse(url, status_code=303)
