"""uvicorn's server, stopped the way `techdebate agents` is: the first Ctrl-C or SIGTERM lets the agents' visits in
progress finish, and a second one cancels them, leaving their notifications pending for next time.

uvicorn alone would not do that. After a first signal it waits for the app's shutdown, which waits for the visits, and
a second Ctrl-C only skips steps it has not reached yet, so the visits would run on. And once it has stopped, it sends
the signal again, which ends the process with SIGTERM's own code rather than 0."""

import contextlib
import signal
import sys
import threading
from collections.abc import Generator
from types import FrameType

import uvicorn
from fastapi import FastAPI

from techdebate.runner import AgentRunner, stopping_message
from techdebate.web.events import EventBus

_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class WebServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, app: FastAPI) -> None:
        super().__init__(config)
        self._app = app
        self.forced = False  # whether a second signal cut the visits in progress short

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        runner: AgentRunner | None = self._app.state.runner
        if not self.should_exit:
            self.should_exit = True
            # uvicorn waits for open connections to close before it shuts the app down, so the live streams of open
            # pages end now.
            events: EventBus = self._app.state.events
            events.close()
            if runner is not None:
                print(stopping_message(runner.in_progress), file=sys.stderr, flush=True)
            return
        self.forced = True
        self.force_exit = True  # nor wait for open connections
        if runner is not None:
            runner.cancel()

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        """As uvicorn's, without sending the signal again once the server has stopped."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        original = {sig: signal.signal(sig, self.handle_exit) for sig in _SIGNALS}
        try:
            yield
        finally:
            for sig, handler in original.items():
                signal.signal(sig, handler)
