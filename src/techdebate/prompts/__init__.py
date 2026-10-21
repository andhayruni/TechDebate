"""Prompt files: <name>.v<N>.md in this package, loaded with load_prompt(name, version=N)."""

from techdebate.prompts.loader import Prompt, load_prompt

__all__ = ["Prompt", "load_prompt"]
