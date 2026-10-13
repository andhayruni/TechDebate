import argparse
import logging

import pytest

from techdebate import cli
from techdebate.errors import ToolError
from techdebate.logging_setup import setup_logging


@pytest.mark.parametrize(
    ("argv", "verbose"),
    [
        (["run", "Rust"], False),
        (["-v", "run", "Rust"], True),
        (["run", "Rust", "-v"], True),
    ],
)
def test_verbose_flag_before_or_after_subcommand(argv: list[str], verbose: bool) -> None:
    assert cli._build_parser().parse_args(argv).verbose is verbose  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(("verbose", "level"), [(True, logging.DEBUG), (False, logging.WARNING)])
def test_setup_logging_only_raises_our_level(verbose: bool, level: int) -> None:
    setup_logging(verbose)
    assert logging.getLogger("techdebate").level == level
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING


def test_techdebate_error_becomes_clean_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(args: argparse.Namespace) -> None:
        raise ToolError("github is down")

    monkeypatch.setattr(cli, "_run", fail)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "Rust"])

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "error: github is down" in err
    assert "Traceback" not in err
