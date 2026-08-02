"""Faceledger command-line application adapter."""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TextIO

from PIL import Image

from faceledger.comparison import (
    ComparisonOutcome,
    ComparisonRequest,
    Diagnostic,
    RecognitionAdapter,
    compare,
)
from faceledger.console import ComparisonConsole, ConsolePresentationFailure

_CLI_MODEL_NAMES = {
    "facenet512": "Facenet512",
    "arcface": "ArcFace",
}


def _match_threshold(value: str) -> float:
    """Parse a finite cosine-distance threshold in the supported range."""

    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number from 0 through 2") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 2.0:
        raise argparse.ArgumentTypeError("must be a finite number from 0 through 2")
    return threshold


def _source_validation_error(source: Path) -> Diagnostic | None:
    """Return a diagnostic when a direct source is not a supported image file."""

    if not source.is_file():
        return Diagnostic(
            severity="error",
            category="input",
            code="source-file-invalid",
            path=source,
            message="The selected source is not a readable regular file.",
        )
    try:
        with Image.open(source) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                return Diagnostic(
                    severity="error",
                    category="input",
                    code="source-format-unsupported",
                    path=source,
                    message=(
                        "The selected source must contain JPEG, PNG, or one-frame "
                        "static WebP image data."
                    ),
                )
            if image.format == "WEBP" and (
                bool(getattr(image, "is_animated", False))
                or getattr(image, "n_frames", 1) > 1
            ):
                return Diagnostic(
                    severity="error",
                    category="input",
                    code="source-webp-animated",
                    path=source,
                    message="The selected source must be a one-frame static WebP.",
                )
            image.verify()
    except (OSError, SyntaxError) as error:
        return Diagnostic(
            severity="error",
            category="input",
            code="source-image-unreadable",
            path=source,
            message=f"The selected source image could not be read: {error}",
        )
    return None


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
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="command",
        required=True,
    )
    comparison_parser = subparsers.add_parser(
        "compare",
        help="compare one source identity with a target face tree",
    )
    comparison_parser.add_argument("source", type=Path)
    comparison_parser.add_argument("target_root", type=Path)
    comparison_parser.add_argument(
        "--model",
        choices=tuple(_CLI_MODEL_NAMES),
        default="facenet512",
    )
    comparison_parser.add_argument("--threshold", type=_match_threshold)
    comparison_parser.add_argument("--no-cache", action="store_true")
    comparison_parser.add_argument("--no-recursive", action="store_true")
    comparison_parser.add_argument("--no-progress", action="store_true")
    return parser


def main(
    arguments: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    recognition: RecognitionAdapter | None = None,
) -> int:
    """Run the shared Faceledger application and return its process status."""

    output = sys.stdout if stdout is None else stdout
    diagnostics = sys.stderr if stderr is None else stderr
    parser = _build_parser()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics):
        try:
            parsed = parser.parse_args(arguments)
        except SystemExit as exit_status:
            return exit_status.code if isinstance(exit_status.code, int) else 1

        if parsed.command == "compare":
            console = ComparisonConsole(
                output,
                diagnostics,
                show_progress=not parsed.no_progress and diagnostics.isatty(),
            )
            try:
                try:
                    source = parsed.source.resolve()
                    source_is_folder = source.is_dir()
                    validation_error = (
                        None if source_is_folder else _source_validation_error(source)
                    )
                    if validation_error is not None:
                        console.diagnostic(validation_error)
                        return console.present(
                            ComparisonOutcome(
                                matches=(),
                                diagnostics=(validation_error,),
                                successful=False,
                            )
                        )
                    outcome = compare(
                        ComparisonRequest(
                            source=None if source_is_folder else source,
                            source_folder=source if source_is_folder else None,
                            target_root=parsed.target_root.resolve(),
                            model_name=_CLI_MODEL_NAMES[parsed.model],
                            threshold=parsed.threshold,
                            single_target_folder=parsed.no_recursive,
                            reuse_cache=not parsed.no_cache,
                        ),
                        recognition,
                        on_diagnostic=console.diagnostic,
                        on_progress=console.progress,
                    )
                except ConsolePresentationFailure:
                    raise
                except Exception as error:  # noqa: BLE001
                    # This public process boundary translates unexpected failures.
                    diagnostic = Diagnostic(
                        severity="error",
                        category="application",
                        code="internal-error",
                        path=None,
                        message=str(error),
                    )
                    console.diagnostic(diagnostic)
                    outcome = ComparisonOutcome(
                        matches=(),
                        diagnostics=(diagnostic,),
                        successful=False,
                    )
                return console.present(outcome)
            except ConsolePresentationFailure as error:
                return console.report_presentation_failure(error)
    raise AssertionError(f"Unsupported parsed command: {parsed.command}")
