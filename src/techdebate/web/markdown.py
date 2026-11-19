"""Posts and comments as markdown. The text comes from models and users, so this is where hostile content would reach
a page, and the renderer is safe by its settings rather than by cleaning up afterwards:

- raw HTML is off, so `<script>` shows as text;
- a link target must be `http` or `https`, so `javascript:`, `data:` and relative links are not made links;
- a bare `http(s)` URL becomes a link, but a file name such as `setup.py` does not;
- images are not loaded: each shows as a link to the image, so a comment cannot make the browser fetch from elsewhere;
- links open in a new tab, with `rel="nofollow noopener noreferrer"`.
"""

import re
from collections.abc import Sequence
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.renderer import RendererHTML
from markdown_it.token import Token
from markdown_it.utils import EnvType, OptionsDict
from markupsafe import Markup, escape

_SCHEMES = frozenset({"http", "https"})
_LINK_REL = "nofollow noopener noreferrer"
_SPACES = re.compile(r"\s+")


def _allowed_link(url: str) -> bool:
    return urlsplit(url.strip()).scheme.lower() in _SCHEMES


def _link_open(self: RendererHTML, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    tokens[idx].attrSet("target", "_blank")
    tokens[idx].attrSet("rel", _LINK_REL)
    return self.renderToken(tokens, idx, options, env)


def _image(self: RendererHTML, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    """An image as a link to it, named by its alt text, or by its URL when it has none."""
    token = tokens[idx]
    src = str(token.attrGet("src") or "")
    alt = self.renderInlineAsText(token.children or [], options, env) or src
    return f'<a href="{escape(src)}" target="_blank" rel="{_LINK_REL}">{escape(alt)}</a>'


def _make() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("linkify")
    md.validateLink = _allowed_link  # pyright: ignore[reportAttributeAccessIssue]
    # Only URLs written with their scheme: "fuzzy" matching would link file names such as setup.py (.py is a country).
    assert md.linkify is not None
    md.linkify.set({"fuzzy_link": False, "fuzzy_email": False, "fuzzy_ip": False})  # pyright: ignore[reportUnknownMemberType]
    md.add_render_rule("link_open", _link_open)
    md.add_render_rule("image", _image)
    return md


_md = _make()


def render_markdown(text: str) -> Markup:
    """The text as safe HTML."""
    # Safe as it is: built with raw HTML off and only http(s) links, and markdown-it escapes all text.
    return Markup(_md.render(text))


def plain_text(text: str) -> str:
    """The text with its markdown taken out, on one line: a link keeps its text and drops its URL, and an image keeps
    its alt text."""
    words: list[str] = []
    for block in _md.parse(text):
        if block.type in ("fence", "code_block"):
            words.append(block.content)
        for token in block.children or []:
            words.append(_inline_text(token))
        words.append(" ")
    return _SPACES.sub(" ", "".join(words)).strip()


def _inline_text(token: Token) -> str:
    if token.type in ("text", "code_inline"):
        return token.content
    if token.type in ("softbreak", "hardbreak"):
        return " "
    if token.type == "image":
        return "".join(_inline_text(child) for child in token.children or [])
    return ""
