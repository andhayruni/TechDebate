import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture() -> Callable[[str], Any]:
    """Load a recorded response body, e.g. load_fixture("github/search_rust.json")."""

    def load(name: str) -> Any:
        return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))

    return load


@pytest.fixture
def load_text_fixture() -> Callable[[str], str]:
    """Load a recorded response body as it is, e.g. load_text_fixture("arxiv/search_vision_transformer.xml")."""

    def load(name: str) -> str:
        return (FIXTURES_DIR / name).read_text(encoding="utf-8")

    return load
