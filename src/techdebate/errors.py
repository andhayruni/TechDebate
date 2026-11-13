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


class BudgetExceeded(TechDebateError):
    """A spending cap was reached."""
