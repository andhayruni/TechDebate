"""Expected failures. The CLI (and later the API) turns these into clean messages instead of tracebacks."""


class TechDebateError(Exception):
    """Base for all expected TechDebate failures."""


class ToolError(TechDebateError):
    """An evidence tool failed."""


class LLMError(TechDebateError):
    """The model provider call failed."""


class PromptError(TechDebateError):
    """A prompt file is unknown, invalid or missing a variable."""


class BudgetExceeded(TechDebateError):
    """A spending cap was reached."""
