from pathlib import Path

import pytest

from techdebate.errors import PromptError
from techdebate.prompts import load_prompt


def _write(base: Path, filename: str, text: str) -> None:
    (base / filename).write_text(text, encoding="utf-8")


def test_renders_variables(tmp_path: Path) -> None:
    _write(tmp_path, "greet.v1.md", "Hello {{ name }}!\n")
    prompt = load_prompt("greet", version=1, base_dir=tmp_path)
    assert prompt.render(name="Rust") == "Hello Rust!\n"


def test_id_name_and_version(tmp_path: Path) -> None:
    _write(tmp_path, "greet.v1.md", "v1")
    _write(tmp_path, "greet.v2.md", "v2")
    prompt = load_prompt("greet", version=2, base_dir=tmp_path)
    assert (prompt.id, prompt.name, prompt.version) == ("greet.v2", "greet", 2)
    assert prompt.render() == "v2"


@pytest.mark.parametrize("text", ["Hello {{ name }}", "{% if name %}hi{% endif %}"])
def test_missing_variable_raises(tmp_path: Path, text: str) -> None:
    _write(tmp_path, "greet.v1.md", text)
    prompt = load_prompt("greet", version=1, base_dir=tmp_path)
    with pytest.raises(PromptError, match=r"greet\.v1 is missing a variable: 'name' is undefined"):
        prompt.render()


def test_unknown_name_raises(tmp_path: Path) -> None:
    _write(tmp_path, "greet.v1.md", "hi")
    with pytest.raises(PromptError, match="unknown prompt: analyst"):
        load_prompt("analyst", version=1, base_dir=tmp_path)


def test_unknown_version_lists_available(tmp_path: Path) -> None:
    _write(tmp_path, "greet.v1.md", "hi")
    _write(tmp_path, "greet.v2.md", "hi")
    with pytest.raises(PromptError, match=r"unknown prompt version: greet\.v3 \(available: v1, v2\)"):
        load_prompt("greet", version=3, base_dir=tmp_path)


@pytest.mark.parametrize("name", ["../greet", "Greet", "greet.v1", ""])
def test_invalid_name_raises(tmp_path: Path, name: str) -> None:
    with pytest.raises(PromptError, match="invalid prompt name"):
        load_prompt(name, version=1, base_dir=tmp_path)


def test_invalid_version_raises(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match="invalid prompt version: 0"):
        load_prompt("greet", version=0, base_dir=tmp_path)


def test_template_syntax_error_raises(tmp_path: Path) -> None:
    _write(tmp_path, "greet.v1.md", "line one\n{% if name %}never closed")
    with pytest.raises(PromptError, match=r"greet\.v1 has a template error on line 2"):
        load_prompt("greet", version=1, base_dir=tmp_path)


def test_default_base_dir_is_the_package() -> None:
    with pytest.raises(PromptError, match="unknown prompt: no_such_prompt"):
        load_prompt("no_such_prompt", version=1)
