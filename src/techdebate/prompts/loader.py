"""Versioned prompt files (<name>.v<N>.md) rendered with Jinja. The ID (e.g. "developer.v1") goes on every LLM call."""

import logging
import re
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable

import jinja2

from techdebate.errors import PromptError

log = logging.getLogger(__name__)

_NAME = re.compile(r"[a-z0-9_]+")
_PROMPT_ID = re.compile(r"([a-z0-9_]+)\.v([1-9][0-9]*)")
# Prompts are plain text, not HTML, so no autoescaping. StrictUndefined makes a missing variable raise.
_env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False, keep_trailing_newline=True)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    template: jinja2.Template

    @property
    def id(self) -> str:
        return f"{self.name}.v{self.version}"

    def render(self, **variables: object) -> str:
        try:
            return self.template.render(**variables)
        except jinja2.UndefinedError as e:
            raise PromptError(f"prompt {self.id} is missing a variable: {e.message}") from e


def load_prompt(name: str, *, version: int, base_dir: Traversable | None = None) -> Prompt:
    """Load <name>.v<version>.md from base_dir, by default the prompts shipped in this package."""
    if not _NAME.fullmatch(name):
        raise PromptError(f"invalid prompt name: {name!r}")
    if version < 1:
        raise PromptError(f"invalid prompt version: {version} (versions start at 1)")

    base = base_dir if base_dir is not None else files("techdebate.prompts")
    available = _versions(base, name)
    if not available:
        raise PromptError(f"unknown prompt: {name}")
    if version not in available:
        listed = ", ".join(f"v{v}" for v in available)
        raise PromptError(f"unknown prompt version: {name}.v{version} (available: {listed})")

    prompt_id = f"{name}.v{version}"
    try:
        template = _env.from_string(base.joinpath(f"{prompt_id}.md").read_text(encoding="utf-8"))
    except jinja2.TemplateSyntaxError as e:
        raise PromptError(f"prompt {prompt_id} has a template error on line {e.lineno}: {e.message}") from e
    log.debug("prompt loaded id=%s", prompt_id)
    return Prompt(name=name, version=version, template=template)


def load_prompt_by_id(prompt_id: str, *, base_dir: Traversable | None = None) -> Prompt:
    """Load a prompt from its ID, e.g. "developer.v1"."""
    match = _PROMPT_ID.fullmatch(prompt_id)
    if match is None:
        raise PromptError(f"invalid prompt ID: {prompt_id!r} (expected <name>.v<N>)")
    return load_prompt(match[1], version=int(match[2]), base_dir=base_dir)


def _versions(base: Traversable, name: str) -> list[int]:
    pattern = re.compile(rf"{re.escape(name)}\.v([1-9][0-9]*)\.md")
    found = (pattern.fullmatch(entry.name) for entry in base.iterdir() if entry.is_file())
    return sorted(int(m.group(1)) for m in found if m)
