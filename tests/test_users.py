import pytest

from techdebate.errors import UserError
from techdebate.models import User, UserKind
from techdebate.roles import ROLE_SPECS, RoleRegistry, RoleSpec
from techdebate.store import Store
from techdebate.users import check_username, shown_name, sync_users

_INVALID = ["", "Andok", "andok-h", "an dok", "andók", "me\n", "a" * 31]


def _roles(*names: str) -> RoleRegistry:
    return RoleRegistry(
        RoleSpec(name=name, display_name=name.title(), question="Why?", tool_names=(), prompt_id="stub.v1")
        for name in names
    )


# check_username


@pytest.mark.parametrize("username", ["me", "andok", "user_01", "0", "a" * 30])
def test_a_name_that_follows_the_rule_is_kept(username: str) -> None:
    assert check_username(username) == username


@pytest.mark.parametrize("username", _INVALID)
def test_a_name_that_breaks_the_rule_is_refused_with_the_rule(username: str) -> None:
    with pytest.raises(UserError, match="use 1 to 30 lowercase letters, digits or underscores"):
        check_username(username)


def test_every_role_name_follows_the_rule() -> None:
    for spec in ROLE_SPECS:
        assert check_username(spec.name) == spec.name


# sync_users


async def test_the_sync_creates_the_agents_and_me(store: Store) -> None:
    users = await sync_users(store, _roles("scientist", "developer"), "me")

    assert list(users.agents) == ["scientist", "developer"]  # registry order
    assert all(agent.kind is UserKind.AGENT for agent in users.agents.values())
    assert [agent.username for agent in users.agents.values()] == ["scientist", "developer"]
    assert (users.me.username, users.me.kind) == ("me", UserKind.REAL)
    assert await store.list_users() == sorted([users.me, *users.agents.values()], key=lambda user: user.username)


async def test_the_sync_can_run_twice(store: Store) -> None:
    roles = _roles("developer")
    first = await sync_users(store, roles, "me")

    second = await sync_users(store, roles, "me")

    assert second == first
    assert len(await store.list_users()) == 2


async def test_a_new_role_gets_its_user_next_to_the_existing_ones(store: Store) -> None:
    first = await sync_users(store, _roles("developer"), "me")

    second = await sync_users(store, _roles("developer", "scientist"), "me")

    assert second.agents["developer"] == first.agents["developer"]
    assert second.me == first.me
    assert await store.find_user("scientist") == second.agents["scientist"]


async def test_a_left_out_agent_keeps_its_user_and_is_not_returned(store: Store) -> None:
    first = await sync_users(store, _roles("developer", "pollster"), "me")

    second = await sync_users(store, _roles("developer"), "me")

    assert list(second.agents) == ["developer"]
    assert await store.find_user("pollster") == first.agents["pollster"]


async def test_a_new_username_makes_a_new_real_user_and_keeps_the_old_one(store: Store) -> None:
    first = await sync_users(store, _roles("developer"), "me")

    second = await sync_users(store, _roles("developer"), "andok")

    assert second.me.username == "andok"
    assert second.me.id != first.me.id
    assert await store.list_users(UserKind.REAL) == [second.me, first.me]


async def test_a_real_user_cannot_take_an_agents_name_and_nothing_is_saved(store: Store) -> None:
    with pytest.raises(UserError, match="username 'developer' belongs to an agent"):
        await sync_users(store, _roles("developer"), "developer")

    assert await store.list_users() == []


async def test_a_new_role_cannot_take_a_real_users_name(store: Store) -> None:
    await sync_users(store, _roles(), "scientist")

    with pytest.raises(UserError, match="the scientist agent cannot be added: a real user already has"):
        await sync_users(store, _roles("developer", "scientist"), "me")

    assert [user.username for user in await store.list_users()] == ["scientist"]


@pytest.mark.parametrize("username", _INVALID)
async def test_the_sync_refuses_a_bad_username_and_saves_nothing(store: Store, username: str) -> None:
    with pytest.raises(UserError, match="is invalid"):
        await sync_users(store, _roles("developer"), username)

    assert await store.list_users() == []


# shown_name


def test_a_real_user_is_shown_by_username() -> None:
    assert shown_name(User(username="andok", kind=UserKind.REAL)) == "andok"


def test_an_agent_whose_role_is_gone_still_carries_the_label() -> None:
    assert shown_name(User(username="retired", kind=UserKind.AGENT)) == "retired [AI]"


@pytest.mark.parametrize("spec", ROLE_SPECS, ids=lambda spec: spec.name)
def test_every_agent_is_shown_by_display_name_with_the_label(spec: RoleSpec) -> None:
    assert shown_name(User(username=spec.name, kind=UserKind.AGENT)) == f"{spec.display_name} [AI]"
