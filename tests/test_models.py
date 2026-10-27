from techdebate.config import Settings
from techdebate.models import Debate, Turn


def test_debate_roundtrip() -> None:
    debate = Debate(topic="Rust", turns=[Turn(analyst="builder", text="...")])
    assert Debate.model_validate_json(debate.model_dump_json()) == debate


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert settings.run_budget_usd <= settings.daily_budget_usd
