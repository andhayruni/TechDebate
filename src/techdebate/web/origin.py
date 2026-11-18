"""Only the app's own pages can write. There is no login yet, so any site open in the browser could send a hidden form
to the app and write as the user in `username`, and every post sets the agents spending. Browsers say where a request
comes from, in `Origin` and `Sec-Fetch-Site`, on every form post, so a write is accepted only when they name the app.

The check runs on every request with an unsafe method, as a dependency of the whole app, so a later form is covered
without a step to forget. Login brings CSRF tokens next to it (F1)."""

from fastapi import Request
from starlette.exceptions import HTTPException

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
REFUSED = "Only the app's own pages can write."


def same_origin(request: Request) -> None:
    """Lets a request through when its method is safe, its `Origin` is the app's own (the scheme with the host and port
    it was sent to), or `Sec-Fetch-Site` is `same-origin`. Refuses everything else with a 403: another site, a
    `null` origin, and a request with neither header."""
    if request.method not in UNSAFE_METHODS:
        return
    own = str(request.base_url).rstrip("/").lower()
    if request.headers.get("origin", "").lower() == own:
        return
    if request.headers.get("sec-fetch-site") == "same-origin":
        return
    raise HTTPException(403, REFUSED)
