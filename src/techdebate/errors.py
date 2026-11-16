"""Expected failures. The CLI (and later the API) turns these into clean messages instead of tracebacks."""


class TechDebateError(Exception):
    """Base for all expected TechDebate failures."""


class ToolError(TechDebateError):
    """An evidence tool failed."""


class NotFoundError(ToolError):
    """An evidence tool worked, and found nothing under that name."""


class LLMError(TechDebateError):
    """The model provider call failed."""


class PromptError(TechDebateError):
    """A prompt file is unknown, invalid or missing a variable."""


class StoreError(TechDebateError):
    """The database refused a write that a user can cause (a username already taken), or is at a version this code
    does not know."""


class UserError(TechDebateError):
    """A username that breaks the rules, or that belongs to a user of the other kind."""


class ForumError(TechDebateError):
    """A write the forum refuses: a post by an agent, blank text, or a reply to a message that does not exist."""


class CommandError(TechDebateError):
    """A command used in a way the discussion does not allow: no post yet, a post ID that names no post or several, or
    an unknown comment number."""


class BudgetExceeded(TechDebateError):
    """A spending cap was reached: what the scope (such as "post") has spent, against its limit."""

    def __init__(self, scope: str, spent_usd: float, limit_usd: float) -> None:
        super().__init__(f"the {scope} has spent ${spent_usd:.4f} of its ${limit_usd:.2f} budget")
        self.scope = scope
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
