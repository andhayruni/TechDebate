from pydantic import HttpUrl

from techdebate.config import Settings
from techdebate.models import Claim, Debate, RelevanceDecision, Turn, TurnKind


def test_claim_citation_flag() -> None:
    cited = Claim(statement="Repo has 10k stars", value=10_000, source_url=HttpUrl("https://github.com/x/y"))
    uncited = Claim(statement="Everyone uses it")
    assert cited.is_cited
    assert not uncited.is_cited


def test_debate_participants_and_roundtrip() -> None:
    debate = Debate(
        topic="Rust",
        decisions=[
            RelevanceDecision(analyst="builder", join=True, reason="Widely used language"),
            RelevanceDecision(analyst="investor", join=False, reason="No funding angle"),
        ],
        turns=[Turn(analyst="builder", round=0, kind=TurnKind.OPENING, text="...")],
    )
    assert debate.participants == ["builder"]
    assert Debate.model_validate_json(debate.model_dump_json()) == debate


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert settings.run_budget_usd <= settings.daily_budget_usd
