"""Record a real API response as a test fixture.

    uv run python tests/record_fixture.py <url> tests/fixtures/<tool>/<case>.json [-H "Name: value" ...]

The request goes through make_http_client, with the same user agent and timeout as the tools. Only the body is saved,
so headers such as a token passed with -H never end up in the file: pretty-printed JSON for a .json file, and the body
as it is for any other file (e.g. arXiv's Atom XML in a .xml file). Keep fixtures small by asking for a small page in
the URL (e.g. per_page=5). Pass --allow-error to keep the body of an error response (e.g. arXiv's error feed).
"""

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from techdebate.config import get_settings
from techdebate.tools import make_http_client


async def fetch(
    client: httpx.AsyncClient, url: str, headers: dict[str, str] | None = None, allow_error: bool = False
) -> httpx.Response:
    response = await client.get(url, headers=headers)
    if not allow_error:
        response.raise_for_status()
    return response


def save(response: httpx.Response, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        out.write_text(json.dumps(response.json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        out.write_text(response.text, encoding="utf-8")


def _header(value: str) -> tuple[str, str]:
    name, sep, val = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected 'Name: value', got {value!r}")
    return name.strip(), val.strip()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Record an API response as a test fixture.")
    parser.add_argument("url")
    parser.add_argument("out", type=Path, help="e.g. tests/fixtures/github/search_rust.json, or .xml to save as is")
    parser.add_argument("-H", "--header", type=_header, action="append", default=[], help='"Name: value"')
    parser.add_argument("--allow-error", action="store_true", help="Save the body of an error response too")
    args = parser.parse_args()

    async with make_http_client(get_settings()) as client:
        try:
            response = await fetch(client, args.url, dict(args.header), args.allow_error)
        except httpx.HTTPError as e:
            raise SystemExit(f"error: {e}") from None
    save(response, args.out)
    print(f"saved {args.out} (HTTP {response.status_code})")


if __name__ == "__main__":
    asyncio.run(_main())
