"""The agent runner: the agents keep running and act on notifications as they arrive.

Each agent has one worker, which makes its visits one after another (AgentLoop.visit) and, with nothing pending, waits
until the forum wakes it or `poll_s` seconds pass, so notifications written by another process are picked up too. The
runner's calls are few (start, stop and wake), so a worker queue or a separate process can later take its place.

Only one process may run the agents on a database, or two would handle the same notifications: `agents_lock` keeps it
to one."""

import asyncio
import fcntl
import logging
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager

from techdebate.config import Settings
from techdebate.errors import CommandError
from techdebate.models import Message, Notification
from techdebate.orchestrator import AgentLoop, Visit
from techdebate.roles import RoleSpec

log = logging.getLogger(__name__)


class AgentRunner:
    """Runs a worker per agent until `stop()`. `on_visit` is called as each visit ends."""

    def __init__(self, loop: AgentLoop, *, poll_s: float, on_visit: Callable[[Visit], None] | None = None) -> None:
        self._loop = loop
        self._poll_s = poll_s
        self._on_visit = on_visit
        # By agent user ID, so a notification wakes the worker of the agent it names.
        self._wakes = {loop.agents[role.name].id: asyncio.Event() for role in loop.roles}
        self._stopping = False
        self._tasks: list[asyncio.Task[None]] = []
        self._in_visit = 0
        self.visits: list[Visit] = []  # every visit, in the order they ended

    @property
    def in_progress(self) -> int:
        """How many workers are in a visit now."""
        return self._in_visit

    async def start(self) -> None:
        """Starts the workers. Notifications already pending are handled at once."""
        self._tasks = [asyncio.create_task(self._work(role), name=f"agent-{role.name}") for role in self._loop.roles]
        log.info("agent runner started agents=%s poll_s=%s", [role.name for role in self._loop.roles], self._poll_s)

    async def stop(self) -> None:
        """Starts no new visit, and returns once the visits in progress have finished, or `cancel()` has ended
        them."""
        self._stopping = True
        for wake in self._wakes.values():
            wake.set()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("agent runner stopped visits=%d", len(self.visits))

    def cancel(self) -> None:
        """Ends the visits in progress at once. Their notifications stay pending, for the next time."""
        self._stopping = True
        for task in self._tasks:
            task.cancel()

    def wake(self, message: Message, notifications: Sequence[Notification]) -> None:
        """A forum listener: wakes the agents the notifications name."""
        for notification in notifications:
            if (wake := self._wakes.get(notification.user_id)) is not None:
                wake.set()

    async def _work(self, role: RoleSpec) -> None:
        wake = self._wakes[self._loop.agents[role.name].id]
        while True:
            # Cleared before the look for notifications, so a wake-up during the look is not lost.
            wake.clear()
            if self._stopping:
                return
            try:
                if await self._loop.has_pending(role):
                    # A stop sets the flag before it wakes, and nothing awaits between this check and the start of
                    # the visit, so no visit starts after a stop.
                    if self._stopping:
                        return
                    self._in_visit += 1
                    try:
                        visit = await self._loop.visit(role)
                    finally:
                        self._in_visit -= 1
                    if visit is not None:
                        self.visits.append(visit)
                        if self._on_visit is not None:
                            self._on_visit(visit)
                        continue
            except Exception:
                # A visit closes its notifications whatever goes wrong in it, so this is something like the database
                # failing. The worker waits a poll interval rather than trying again at once.
                log.exception("agent worker failed agent=%s", role.name)
            try:
                async with asyncio.timeout(self._poll_s):
                    await wake.wait()
            except TimeoutError:
                pass


def lock_path(settings: Settings) -> str:
    return f"{settings.db_path}.agents.lock"


@contextmanager
def agents_lock(settings: Settings) -> Generator[None]:
    """Held while the agents run on the database at db_path. The OS releases it when the process ends, even after a
    crash. Raises CommandError while another process holds it."""
    with open(lock_path(settings), "a") as file:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CommandError(
                "the agents are already running on this database (techdebate agents, run or the web app): stop it first"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
