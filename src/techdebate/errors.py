"""Expected failures. The CLI (and later the API) turns these into clean messages instead of tracebacks."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from techdebate.llm.base import Usage


class TechDebateError(Exception):
    """Base for all expected TechDebate failures."""


class ToolError(TechDebateError):
    """An evidence tool failed."""


class NotFoundError(ToolError):
    """An evidence tool worked, and found nothing under that name."""


class LLMError(TechDebateError):
    """The model provider call failed. When the provider answered (a refusal, output cut off at max_tokens, or output
    that does not fit the schema), the call is paid for, and the error carries the model that answered, its usage, its
    cost and its text, so the call can be stored. Otherwise it carries at most the model that was requested."""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        usage: "Usage | None" = None,
        cost_usd: float = 0.0,
        text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.usage = usage
        self.cost_usd = cost_usd
        self.text = text


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
    """A command used in a way the discussion does not allow: no post yet, a post ID that names no post or several, an
    unknown comment number, or the agents started while another process runs them."""


def usd_limit(limit_usd: float) -> str:
    """A budget's limit in dollars: with cents, and with more digits only when it has them, so $0.015 is not shown as
    $0.01."""
    cents = f"{limit_usd:.2f}"
    if float(cents) == round(limit_usd, 6):
        return f"${cents}"
    return f"${limit_usd:.6f}".rstrip("0")


# What each budget scope's message says, by its name. A scope not listed here is named as it is.
_SPENT = {
    "post": "the post has spent {spent} of its {limit} budget",
    "agent": "the agent has spent {spent} of its {limit} budget for today",
    "day": "the agents have spent {spent} of their {limit} budget for today",
}


class BudgetExceeded(TechDebateError):
    """A spending cap was reached: what the scope ("post", "agent" or "day") has spent, against its limit."""

    def __init__(self, scope: str, spent_usd: float, limit_usd: float) -> None:
        template = _SPENT.get(scope, f"the {scope} has spent {{spent}} of its {{limit}} budget")
        super().__init__(template.format(spent=f"${spent_usd:.4f}", limit=usd_limit(limit_usd)))
        self.scope = scope
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
