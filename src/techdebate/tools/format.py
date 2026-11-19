"""Evidence as text, for the terminal and for an analyst's prompt."""

from techdebate.models import Evidence


def _format_value(value: float) -> str:
    return f"{int(value):,}" if value.is_integer() else f"{value:,.2f}"


def format_evidence(items: list[Evidence]) -> str:
    """One numbered block per Evidence, separated by blank lines."""
    blocks: list[str] = []
    for n, item in enumerate(items, start=1):
        title = f"[{n}] {item.title}"
        if item.value is not None:
            title += f"  (value: {_format_value(item.value)})"
        blocks.append(f"{title}\n    {item.summary}\n    {item.url}")
    return "\n\n".join(blocks)
