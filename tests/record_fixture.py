"""Record a real API response as a test fixture.

    uv run python tests/record_fixture.py <url> tests/fixtures/<tool>/<case>.json [-H "Name: value" ...]

The request goes through make_http_client, with the same user agent and timeout as the tools. Only the JSON body is
saved, pretty-printed, so headers such as a token passed with -H never end up in the file. Keep fixtures small by
asking for a small page in the URL (e.g. per_page=5).
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from techdebate.config import get_settings
from techdebate.tools import make_http_client


async def fetch(client: httpx.AsyncClient, url: str, headers: dict[str, str] | None = None) -> Any:
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def save(body: Any, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _header(value: str) -> tuple[str, str]:
    name, sep, val = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected 'Name: value', got {value!r}")
    return name.strip(), val.strip()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Record a JSON API response as a test fixture.")
    parser.add_argument("url")
    parser.add_argument("out", type=Path, help="e.g. tests/fixtures/github/search_rust.json")
    parser.add_argument("-H", "--header", type=_header, action="append", default=[], help='"Name: value"')
    args = parser.parse_args()

    async with make_http_client(get_settings()) as client:
        try:
            body = await fetch(client, args.url, dict(args.header))
        except httpx.HTTPError as e:
            raise SystemExit(f"error: {e}") from None
    save(body, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    asyncio.run(_main())
