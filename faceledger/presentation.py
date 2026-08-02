"""Human-readable comparison result and artifact presentation."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

from faceledger.comparison import ComparisonOutcome, Diagnostic


@dataclass(frozen=True)
class ComparisonArtifactRequest:
    """Optional destinations selected for one comparison run."""

    result_path: Path | None = None
    log_path: Path | None = None


def _write_artifact(path: Path, text: str) -> None:
    """Overwrite one regular UTF-8 artifact without creating its parent."""

    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"Artifact parent directory does not exist: {path.parent}"
        )
    if (path.exists() or path.is_symlink()) and not path.is_file():
        raise OSError(f"Artifact destination is not a regular file: {path}")
    path.write_text(text, encoding="utf-8")


def render_matches(outcome: ComparisonOutcome) -> str:
    """Render candidate matches for standard output."""

    if not outcome.successful:
        return ""
    if not outcome.matches:
        return "No matches found\n"
    identity_width = max(
        len("Identity"),
        *(len(str(match.identity_path)) for match in outcome.matches),
    )
    identity_column_width = identity_width + 2
    lines = [f"{'Rank':<6}{'Identity':<{identity_column_width}}Cosine distance\n"]
    lines.extend(
        f"{rank:<6}{match.identity_path!s:<{identity_column_width}}"
        f"{match.cosine_distance:.6f}\n"
        for rank, match in enumerate(outcome.matches, start=1)
    )
    return "".join(lines)


def render_comparison_result(outcome: ComparisonOutcome) -> str:
    """Render the complete resolved human-readable comparison result."""

    if not outcome.successful:
        return ""
    metadata = outcome.metadata
    if metadata is None:
        raise ValueError("A successful comparison must have operation metadata.")
    return (
        f"Source: {metadata.source}\n"
        f"Target root: {metadata.target_root}\n"
        f"Model: {metadata.model_name}\n"
        f"Threshold: {metadata.threshold:.6f}\n\n"
        f"{render_matches(outcome)}"
    )


def render_diagnostic(diagnostic: Diagnostic) -> str:
    """Format one structured diagnostic for human-readable output."""

    location = f" {diagnostic.path}" if diagnostic.path is not None else ""
    return (
        f"{diagnostic.severity.upper()} "
        f"[{diagnostic.category}:{diagnostic.code}]"
        f"{location}: {diagnostic.message}\n"
    )


def render_comparison_log(outcome: ComparisonOutcome) -> str:
    """Render one UTF-8 troubleshooting record without ranked results."""

    metadata = outcome.metadata
    status = (
        "cancelled"
        if not outcome.complete
        else "successful"
        if outcome.successful
        else "unsuccessful"
    )
    metadata_text = (
        f"Source: {metadata.source}\n"
        f"Target root: {metadata.target_root}\n"
        f"Model: {metadata.model_name}\n"
        f"Threshold: {metadata.threshold:.6f}\n"
        if metadata is not None
        else "Operation metadata: unavailable\n"
    )
    diagnostics = "".join(render_diagnostic(item) for item in outcome.diagnostics)
    return (
        metadata_text
        + f"Status: {status}\n"
        + f"Target identities compared: {outcome.target_identities_compared}\n"
        + f"Candidate matches: {len(outcome.matches)}\n"
        + f"Diagnostics: {len(outcome.diagnostics)}\n"
        + diagnostics
    )


def write_comparison_artifacts(
    outcome: ComparisonOutcome,
    request: ComparisonArtifactRequest,
) -> ComparisonOutcome:
    """Write selected artifacts and append any failure diagnostics."""

    if request.result_path is None and request.log_path is None:
        return outcome

    current = outcome
    if request.result_path is not None and outcome.successful and outcome.complete:
        try:
            _write_artifact(
                request.result_path,
                render_comparison_result(outcome),
            )
        except OSError as error:
            current = replace(
                current,
                diagnostics=current.diagnostics
                + (
                    Diagnostic(
                        severity="error",
                        category="output",
                        code="result-artifact-write-failed",
                        path=request.result_path,
                        message=str(error),
                    ),
                ),
                successful=False,
            )

    if request.log_path is not None:
        try:
            _write_artifact(
                request.log_path,
                render_comparison_log(current),
            )
        except OSError as error:
            current = replace(
                current,
                diagnostics=current.diagnostics
                + (
                    Diagnostic(
                        severity="error",
                        category="output",
                        code="log-artifact-write-failed",
                        path=request.log_path,
                        message=str(error),
                    ),
                ),
                successful=False,
            )
    return current


def present_comparison(
    outcome: ComparisonOutcome,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Present a retained outcome through the terminal console component."""

    from faceledger.console import ComparisonConsole

    console = ComparisonConsole(stdout, stderr)
    for diagnostic in outcome.diagnostics:
        console.diagnostic(diagnostic)
    return console.present(outcome)
