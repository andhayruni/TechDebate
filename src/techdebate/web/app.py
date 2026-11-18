"""The web app: the pages a person reads and writes the discussions in, on my own machine.

`create_app` builds the app around one session (the store, the users and the forum), opened when the app starts and
closed when it stops. Pages get it, and the settings, through the dependencies in `deps`. There is no login yet, so the
app answers only requests addressed to this machine, and everything written from it is written by the user in
`username`.
"""

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from http import HTTPStatus
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from starlette.exceptions import HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from techdebate.config import Settings, get_settings
from techdebate.errors import TechDebateError
from techdebate.llm import AnthropicLLM, LLMClient
from techdebate.orchestrator import AgentLoop, Visit
from techdebate.runner import AgentRunner, agents_lock
from techdebate.session import open_session
from techdebate.web.origin import same_origin
from techdebate.web.routes import router
from techdebate.web.templating import templates

log = logging.getLogger(__name__)

# The app listens on 127.0.0.1 only. A request addressed to any other name is refused, so a web page in my browser
# that points its own domain at this machine (DNS rebinding) cannot read the app.
HOST = "127.0.0.1"
ALLOWED_HOSTS = [HOST, "localhost"]

# What an error page says when the error itself has nothing written for a reader.
_HTTP_MESSAGES = {
    HTTPStatus.NOT_FOUND: "There is no page here.",
    HTTPStatus.UNPROCESSABLE_ENTITY: "The request has a missing or invalid value.",
    HTTPStatus.INTERNAL_SERVER_ERROR: "Something went wrong on our side. The details are in the app's log.",
}


def create_app(
    settings: Settings | None = None,
    *,
    agents: bool = True,
    llm: LLMClient | None = None,
    on_visit: Callable[[Visit], None] | None = None,
) -> FastAPI:
    """The app, on `settings` or the ones from the environment. `uvicorn --factory` calls it with none.

    With `agents`, the app runs the agents while it runs: it takes the agents' lock, so no other process runs them on
    the database, and starts the runner, which the forum wakes when a message is written. `llm` is the model the agents
    call (AnthropicLLM without it), and `on_visit` hears of each visit as it ends."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[dict[str, Any]]:
        async with open_session(settings) as session, AsyncExitStack() as stack:
            if agents:
                stack.enter_context(agents_lock(settings))
                loop = AgentLoop(
                    session.store,
                    session.forum,
                    session.agents,
                    llm=llm or AnthropicLLM(settings),
                    roles=session.roles,
                    tools=session.tools,
                    settings=settings,
                )
                runner = AgentRunner(loop, poll_s=settings.agent_poll_s, on_visit=on_visit)
                session.forum.subscribe(runner.wake)
                await runner.start()
                # Stopped before the lock is released and the session closed: the visits in progress finish first.
                stack.push_async_callback(runner.stop)
                app.state.runner = runner
            log.info("web app started db=%s agents=%s", settings.db_path, agents)
            yield {"session": session}
        app.state.runner = None
        log.info("web app stopped")

    # No API docs: the pages are HTML, and a JSON API is future work (F4).
    app = FastAPI(
        title="TechDebate",
        lifespan=lifespan,
        # Every route, so every later form too, is behind the origin check.
        dependencies=[Depends(same_origin)],
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.agents = agents
    app.state.runner = None  # the agent runner while the app runs, with `agents`
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.mount("/static", StaticFiles(packages=[("techdebate.web", "static")]), name="static")
    app.add_exception_handler(TechDebateError, _techdebate_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    # Registered for Exception, Starlette calls it from its outermost middleware and then raises the error again, so
    # the server logs its traceback.
    app.add_exception_handler(Exception, _unexpected_error)

    @app.get("/health")
    async def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    app.include_router(router)
    return app


# Error pages


def error_page(request: Request, status: int, message: str) -> Response:
    """A page in the base layout with the status and a message written for a reader."""
    context = {"status": status, "phrase": HTTPStatus(status).phrase, "message": message}
    return templates.TemplateResponse(request, "error.html", context, status_code=status)


async def _techdebate_error(request: Request, exc: Exception) -> Response:
    """An expected failure: its message, the line the CLI would print."""
    log.debug("request failed path=%s", request.url.path, exc_info=exc)
    return error_page(request, HTTPStatus.BAD_REQUEST, str(exc))


async def _http_error(request: Request, exc: Exception) -> Response:
    """An HTTP error, such as an unknown URL, with its status. A detail written for the error is shown; the default one,
    which only repeats the status, is not."""
    assert isinstance(exc, HTTPException)
    detail = exc.detail if exc.detail != HTTPStatus(exc.status_code).phrase else ""
    response = error_page(request, exc.status_code, detail or _HTTP_MESSAGES.get(HTTPStatus(exc.status_code), ""))
    if exc.headers:  # such as Allow on a 405
        response.headers.update(exc.headers)
    return response


async def _validation_error(request: Request, exc: Exception) -> Response:
    return error_page(request, HTTPStatus.UNPROCESSABLE_ENTITY, _HTTP_MESSAGES[HTTPStatus.UNPROCESSABLE_ENTITY])


async def _unexpected_error(request: Request, exc: Exception) -> Response:
    """A bug. The page says only that something went wrong: the traceback can show local paths and settings, so it goes
    to the log."""
    return error_page(request, HTTPStatus.INTERNAL_SERVER_ERROR, _HTTP_MESSAGES[HTTPStatus.INTERNAL_SERVER_ERROR])
