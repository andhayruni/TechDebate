"""One-time logging setup, called from the CLI.

Modules log with `log = logging.getLogger(__name__)` and write key=value messages, e.g.
`log.warning("tool failed tool=%s error=%s", name, exc)`.
"""

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    # Logs go to stderr so the debate printed on stdout stays clean.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Only our own loggers get louder; libraries like httpx (which logs every request at INFO) stay at WARNING.
    logging.getLogger("techdebate").setLevel(logging.DEBUG if verbose else logging.WARNING)
