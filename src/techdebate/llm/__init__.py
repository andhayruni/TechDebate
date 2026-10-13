"""LLM layer: one small interface (LLMClient) and a scripted fake for tests."""

from techdebate.llm.base import LLMClient, LLMRequest, LLMResponse, Message, ModelTier, Usage
from techdebate.llm.fake import FakeLLM

__all__ = ["FakeLLM", "LLMClient", "LLMRequest", "LLMResponse", "Message", "ModelTier", "Usage"]
