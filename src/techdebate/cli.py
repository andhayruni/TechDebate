"""Command-line entry point. `techdebate run "<topic>"` arrives with the thin slice."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="techdebate")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a debate on a technology")
    run.add_argument("topic")
    args = parser.parse_args()

    if args.command == "run":
        raise SystemExit("Debate loop not implemented yet.")
