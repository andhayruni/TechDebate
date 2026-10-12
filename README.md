# TechDebate

A panel of AI analysts that debates whether a technology is promising. Each analyst
gathers evidence from free public sources and cites it.

## Development

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run ruff check . && uv run pyright && uv run pytest
uv run techdebate run "Artificial intelligence"
```
