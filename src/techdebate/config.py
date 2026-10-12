"""Runtime settings, read from the environment (TECHDEBATE_*) or a local .env file."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TECHDEBATE_", env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    github_token: SecretStr | None = Field(default=None, validation_alias="GITHUB_TOKEN")

    # Cheap model decides whether an analyst joins; strong model writes the debate turns.
    relevance_model: str = "claude-haiku-5-5"
    debate_model: str = "claude-sonnet-5-5"

    run_budget_usd: float = 0.50
    daily_budget_usd: float = 5.00

    db_path: str = "techdebate.db"
    rebuttal_rounds: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
