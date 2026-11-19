from pydantic import HttpUrl

from techdebate.models import Evidence
from techdebate.tools import format_evidence


def _item(name: str, value: float | None = None) -> Evidence:
    return Evidence(
        source="stub",
        query="q",
        url=HttpUrl(f"https://example.com/{name}"),
        title=name,
        summary=f"{name} summary",
        value=value,
    )


def test_numbers_the_blocks_and_shows_the_value_when_set() -> None:
    text = format_evidence([_item("a", 1234567.0), _item("b"), _item("c", 0.125)])

    assert text == (
        "[1] a  (value: 1,234,567)\n    a summary\n    https://example.com/a\n\n"
        "[2] b\n    b summary\n    https://example.com/b\n\n"
        "[3] c  (value: 0.12)\n    c summary\n    https://example.com/c"
    )


def test_no_evidence_is_an_empty_string() -> None:
    assert format_evidence([]) == ""
