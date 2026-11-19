"""Roles are data, not subclasses. Each new role adds one entry to ROLE_SPECS."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from techdebate.tools import ToolRegistry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoleSpec:
    name: str
    display_name: str
    question: str  # the role's guiding question
    tool_names: tuple[str, ...]
    prompt_id: str  # the prompt the role writes its turns with


ROLE_SPECS: tuple[RoleSpec, ...] = (
    RoleSpec(
        name="developer",
        display_name="Developer",
        question="Are developers building with it and using it?",
        tool_names=("github_search", "github_repo", "pypi"),
        prompt_id="developer.v1",
    ),
    RoleSpec(
        name="scientist",
        display_name="Scientist",
        question="Are researchers publishing on it, and is that growing?",
        tool_names=("arxiv",),
        prompt_id="scientist.v1",
    ),
    RoleSpec(
        name="pollster",
        display_name="Pollster",
        question="Is the public interested, and is interest growing?",
        tool_names=("youtube",),
        prompt_id="pollster.v1",
    ),
)


def display_name(role_name: str) -> str:
    """The role's display name, or the name itself for a role no longer in ROLE_SPECS."""
    return next((spec.display_name for spec in ROLE_SPECS if spec.name == role_name), role_name)


class RoleRegistry:
    def __init__(self, roles: Iterable[RoleSpec] = ()) -> None:
        self._roles: dict[str, RoleSpec] = {}
        for role in roles:
            self.register(role)

    def register(self, role: RoleSpec) -> None:
        if role.name in self._roles:
            raise ValueError(f"role already registered: {role.name}")
        self._roles[role.name] = role

    def roles(self) -> list[RoleSpec]:
        """All roles, in the order they were registered."""
        return list(self._roles.values())


def default_roles(tools: ToolRegistry, specs: Iterable[RoleSpec] = ROLE_SPECS) -> RoleRegistry:
    """Every role whose tools are all available. Any other role is left out with a warning."""
    available = {tool.name for tool in tools.tools()}
    registry = RoleRegistry()
    for spec in specs:
        missing = [name for name in spec.tool_names if name not in available]
        if missing:
            log.warning("role left out role=%s missing_tools=%s", spec.name, ",".join(missing))
            continue
        registry.register(spec)
    return registry
