"""LLM layer: one small interface (LLMClient), the Anthropic client, a cost meter, a client that stores every call,
and a scripted fake for tests."""

from techdebate.llm.anthropic_client import AnthropicLLM
from techdebate.llm.base import LLMClient, LLMRequest, LLMResponse, Message, ModelTier, Usage
from techdebate.llm.fake import FakeLLM
from techdebate.llm.meter import CostMeter
from techdebate.llm.recording import RecordingLLM

__all__ = [
    "AnthropicLLM",
    "CostMeter",
    "FakeLLM",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "ModelTier",
    "RecordingLLM",
    "Usage",
]
