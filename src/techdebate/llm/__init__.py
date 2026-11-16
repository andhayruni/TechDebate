"""LLM layer: one small interface (LLMClient), the Anthropic client, a client that stores every call, one that
refuses calls once a budget is used up, and a scripted fake for tests."""

from techdebate.llm.anthropic_client import AnthropicLLM
from techdebate.llm.base import LLMClient, LLMRequest, LLMResponse, Message, ModelTier, Usage
from techdebate.llm.budget import Budget, BudgetScope, post_scopes
from techdebate.llm.fake import FakeLLM
from techdebate.llm.recording import RecordingLLM

__all__ = [
    "AnthropicLLM",
    "Budget",
    "BudgetScope",
    "FakeLLM",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "ModelTier",
    "RecordingLLM",
    "Usage",
    "post_scopes",
]
