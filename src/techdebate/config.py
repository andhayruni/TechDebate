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

    # Cheap tier: relevance checks and query planning. Strong tier: the debate turns.
    cheap_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-sonnet-5-5"
    # Both models think by default, and thinking counts against max_tokens; low effort keeps cheap calls short.
    cheap_effort: Effort = "low"
    strong_effort: Effort = "high"
    model_prices: dict[str, ModelPrice] = Field(default_factory=_default_prices)

    # The SDK retries rate limits, server and connection errors with backoff.
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    # Evidence tools share one HTTP client. Public APIs (Wikimedia, GitHub) ask for a descriptive user agent.
    http_timeout_s: float = 20.0
    http_user_agent: str = Field(default_factory=_default_user_agent)

    run_budget_usd: float = 0.50
    daily_budget_usd: float = 5.00

    db_path: str = "techdebate.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
