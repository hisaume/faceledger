"""Faceledger command-line application adapter."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from importlib.metadata import version
from typing import TextIO


def _build_parser() -> argparse.ArgumentParser:
    """Build the discoverable top-level Faceledger parser."""

    parser = argparse.ArgumentParser(
        prog="faceledger",
        description="Find plausible candidate matches in a local face tree.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('faceledger')}",
    )
    parser.add_subparsers(dest="command", metavar="command", required=True)
    return parser


def main(
    arguments: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the shared Faceledger application and return its process status."""

    output = sys.stdout if stdout is None else stdout
    diagnostics = sys.stderr if stderr is None else stderr
    parser = _build_parser()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics):
        try:
            parser.parse_args(arguments)
        except SystemExit as exit_status:
            return exit_status.code if isinstance(exit_status.code, int) else 1
    return 0
