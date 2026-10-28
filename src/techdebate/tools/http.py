"""The HTTP client the evidence tools share.

The caller owns the client: it opens one per run (`async with make_http_client(settings) as client:`) and passes it
to every tool, so they share a connection pool. Tools never create or close it.
"""

import httpx

from techdebate.config import Settings


def make_http_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=settings.http_timeout_s,
        headers={"User-Agent": settings.http_user_agent},
        follow_redirects=True,
    )
