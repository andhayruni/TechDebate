"""The pages: the feed, and a post with its threads. They read through the `Store`, and need no AI."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, Response
from starlette.exceptions import HTTPException

from techdebate.session import Session
from techdebate.tree import Tree
from techdebate.web.deps import get_session
from techdebate.web.pages import feed, post_page
from techdebate.web.templating import templates

router = APIRouter(default_response_class=HTMLResponse)

CurrentSession = Annotated[Session, Depends(get_session)]


@router.get("/", name="home")
async def home(request: Request, session: CurrentSession) -> Response:
    """The feed: the posts, newest first."""
    store = session.store
    users = {user.id: user for user in await store.list_users()}
    items = feed(await store.list_post_summaries(), users)
    return templates.TemplateResponse(request, "feed.html", {"posts": items})


@router.get("/posts/{post_id}", name="post")
async def post(request: Request, post_id: str, session: CurrentSession) -> Response:
    """A post and its threads. The ID is the post's full ID."""
    store = session.store
    found = await store.get_post(post_id)
    if found is None:
        raise HTTPException(404, "There is no post here.")
    users = {user.id: user for user in await store.list_users()}
    return templates.TemplateResponse(request, "post.html", {"page": post_page(Tree.of(found), users)})
