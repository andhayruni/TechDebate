"""The interface between our code and a model provider. Requests name a tier, never a model."""

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel


class ModelTier(StrEnum):
    CHEAP = "cheap"
    STRONG = "strong"


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    tier: ModelTier
    system: str
    messages: list[Message]
    output_schema: type[BaseModel] | None = None
    max_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class LLMResponse(BaseModel):
    text: str
    parsed: BaseModel | None = None  # set when the request had an output_schema
    usage: Usage
    cost_usd: float
    model: str


class LLMClient(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
