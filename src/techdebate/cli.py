"""Command-line entry point. `techdebate run "<topic>"` arrives with the thin slice."""

import argparse
import logging
import sys

from techdebate.errors import TechDebateError
from techdebate.logging_setup import setup_logging

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="techdebate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs and full tracebacks")
    # Same flag on subcommands; SUPPRESS keeps a `-v` given before the subcommand from being reset.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", parents=[common], help="Run a debate on a technology")
    run.add_argument("topic")
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.command == "run":
        raise SystemExit("Debate loop not implemented yet.")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        _run(args)
    except TechDebateError as e:
        log.debug("command failed command=%s", args.command, exc_info=True)
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
