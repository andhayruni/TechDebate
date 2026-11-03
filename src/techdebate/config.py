"""Runtime settings, read from the environment (TECHDEBATE_*) or a local .env file."""

from functools import lru_cache
from importlib.metadata import version
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort: TypeAlias = Literal["low", "medium", "high", "xhigh", "max"]


class ModelPrice(BaseModel):
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float


# Prices as of 2026-10.
def _default_prices() -> dict[str, ModelPrice]:
    return {
        # Haiku's rate is for prompts up to 100K tokens ($0.50 / $2.50 beyond); our prompts stay well under.
        "claude-haiku-5-5": ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50),
        "claude-sonnet-5-5": ModelPrice(input_per_mtok=2.00, output_per_mtok=10.00),
    }


def _default_user_agent() -> str:
    return f"techdebate/{version('techdebate')} (+https://github.com/andhayruni/TechDebate)"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TECHDEBATE_", env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    github_token: SecretStr | None = Field(default=None, validation_alias="GITHUB_TOKEN")
    youtube_api_key: SecretStr | None = Field(default=None, validation_alias="YOUTUBE_API_KEY")

    # Cheap tier: speak decisions and query plans. Strong tier: the debate turns.
    cheap_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-sonnet-5-5"
    # Both models think by default, and thinking counts against max_tokens; low effort keeps cheap calls short.
    cheap_effort: Effort = "low"
    strong_effort: Effort = "high"
    model_prices: dict[str, ModelPrice] = Field(default_factory=_default_prices)

    # Thinking counts against this too, so it leaves room for both.
    write_max_tokens: int = Field(default=4096, ge=1)

    # The SDK retries rate limits, server and connection errors with backoff.
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    # Evidence tools share one HTTP client. Public APIs (GitHub, PyPI, arXiv, YouTube) ask for a descriptive user agent.
    http_timeout_s: float = 20.0
    http_user_agent: str = Field(default_factory=_default_user_agent)

    # GitHub: how many top repositories github_search returns, and the activity window github_repo counts over.
    github_top_n: int = Field(default=5, ge=1, le=100)
    github_activity_days: int = Field(default=90, ge=1)

    # arXiv: how many top papers to return, and how many years the publication trend counts (the current year to date
    # plus the ones before it; 0 skips it). arXiv asks for at most one request every 3 seconds and the trend costs one
    # request per year, so each year adds about 3 seconds to a run.
    arxiv_top_n: int = Field(default=5, ge=1, le=100)
    arxiv_trend_years: int = Field(default=5, ge=0, le=20)
    arxiv_request_interval_s: float = Field(default=3.0, ge=0)

    # YouTube: how many top videos to return. The API allows at most 50 per search.
    youtube_top_n: int = Field(default=10, ge=1, le=50)

    run_budget_usd: float = 0.50
    daily_budget_usd: float = 5.00

    db_path: str = "techdebate.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
