"""The interface every evidence tool implements. Tools know nothing about debates or analysts."""

from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from techdebate.models import Evidence


class ToolInput(BaseModel):
    """Base for a tool's input model. Its JSON schema is what an LLM or MCP client fills in, so unknown fields fail."""

    model_config = ConfigDict(extra="forbid")


InputT = TypeVar("InputT", bound=ToolInput)


class Tool(Protocol[InputT]):
    # Read-only properties, so a tool can implement them as plain class attributes (`name = "github"`).
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...  # one short line; also tells a role that hasn't joined what the tool finds

    @property
    def input_model(self) -> type[InputT]: ...

    async def run(self, input: InputT) -> list[Evidence]: ...
