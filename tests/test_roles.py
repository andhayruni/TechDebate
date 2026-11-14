import logging

import pytest

from techdebate.config import Settings
from techdebate.models import Evidence
from techdebate.roles import ROLE_SPECS, RoleRegistry, RoleSpec, default_roles, display_name
from techdebate.tools import ToolInput, ToolRegistry, default_tools, make_http_client


class _StubInput(ToolInput):
    query: str


class _StubTool:
    description = "Finds nothing"
    input_model = _StubInput

    def __init__(self, name: str) -> None:
        self.name = name

    async def run(self, input: _StubInput) -> list[Evidence]:
        return []


def _tools(*names: str) -> ToolRegistry:
    return ToolRegistry(_StubTool(name) for name in names)


def _role(name: str, *tool_names: str) -> RoleSpec:
    return RoleSpec(name=name, display_name=name.title(), question="Why?", tool_names=tool_names, prompt_id="stub.v1")


def _settings(youtube_api_key: str | None) -> Settings:
    return Settings(_env_file=None, YOUTUBE_API_KEY=youtube_api_key)  # pyright: ignore[reportCallIssue]


# display_name


def test_display_name_comes_from_the_role_spec() -> None:
    assert display_name("developer") == "Developer"


def test_display_name_of_an_unknown_role_is_the_name_itself() -> None:
    assert display_name("retired") == "retired"


# RoleRegistry


def test_registry_lists_roles_in_registration_order() -> None:
    a, b, c = _role("b"), _role("a"), _role("c")
    registry = RoleRegistry([a, b])
    registry.register(c)

    assert registry.roles() == [a, b, c]


def test_duplicate_name_is_rejected() -> None:
    registry = RoleRegistry([_role("developer")])
    with pytest.raises(ValueError, match="already registered: developer"):
        registry.register(_role("developer"))


# default_roles


_DEVELOPER = RoleSpec(
    name="developer",
    display_name="Developer",
    question="Are developers building with it and using it?",
    tool_names=("github_search", "github_repo", "pypi"),
    prompt_id="developer.v1",
)
_SCIENTIST = RoleSpec(
    name="scientist",
    display_name="Scientist",
    question="Are researchers publishing on it, and is that growing?",
    tool_names=("arxiv",),
    prompt_id="scientist.v1",
)
_POLLSTER = RoleSpec(
    name="pollster",
    display_name="Pollster",
    question="Is the public interested, and is interest growing?",
    tool_names=("youtube",),
    prompt_id="pollster.v1",
)


async def test_all_three_agents_are_registered_in_order_with_a_youtube_key() -> None:
    settings = _settings(youtube_api_key="key")
    async with make_http_client(settings) as client:
        registry = default_roles(default_tools(settings, client))

    assert registry.roles() == [_DEVELOPER, _SCIENTIST, _POLLSTER]


async def test_without_a_youtube_key_only_the_pollster_is_left_out(caplog: pytest.LogCaptureFixture) -> None:
    settings = _settings(youtube_api_key=None)
    async with make_http_client(settings) as client:
        registry = default_roles(default_tools(settings, client))

    assert registry.roles() == [_DEVELOPER, _SCIENTIST]  # arXiv needs no key
    assert [record.getMessage() for record in caplog.records] == ["role left out role=pollster missing_tools=youtube"]


@pytest.mark.parametrize(("name", "shown"), [("scientist", "Scientist"), ("pollster", "Pollster")])
def test_display_names_of_the_new_agents(name: str, shown: str) -> None:
    assert display_name(name) == shown


async def test_every_role_names_tools_that_exist(caplog: pytest.LogCaptureFixture) -> None:
    settings = _settings(youtube_api_key="key")  # every tool that needs a key has one
    async with make_http_client(settings) as client:
        registry = default_roles(default_tools(settings, client))

    assert registry.roles() == list(ROLE_SPECS)
    assert caplog.records == []


def test_developer_is_left_out_when_a_tool_is_missing(caplog: pytest.LogCaptureFixture) -> None:
    registry = default_roles(_tools("github_search", "github_repo"), [_DEVELOPER])

    assert registry.roles() == []
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "role left out role=developer missing_tools=pypi"


def test_a_left_out_role_names_all_its_missing_tools_and_the_others_are_kept(caplog: pytest.LogCaptureFixture) -> None:
    kept, left_out = _role("scientist", "arxiv"), _role("pollster", "youtube", "trends", "arxiv")

    registry = default_roles(_tools("arxiv"), [left_out, kept])

    assert registry.roles() == [kept]
    assert [record.getMessage() for record in caplog.records] == [
        "role left out role=pollster missing_tools=youtube,trends"
    ]
