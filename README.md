# TechDebate

A panel of AI analysts that debates whether a technology is promising. Each analyst
gathers evidence from free public sources and cites it.

## Development

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run ruff check . && uv run pyright && uv run pytest
uv run techdebate ask "What is Rust?"   # smoke check: answer + cost
uv run techdebate evidence github_repo "rust-lang/rust"   # run one evidence tool
uv run techdebate run "Artificial intelligence"   # add --no-delay to skip the reading delays
```
