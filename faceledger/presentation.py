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


def _render_diagnostic(diagnostic: Diagnostic) -> str:
    location = f" {diagnostic.path}" if diagnostic.path is not None else ""
    return (
        f"{diagnostic.severity.upper()} "
        f"[{diagnostic.category}:{diagnostic.code}]"
        f"{location}: {diagnostic.message}\n"
    )


def render_comparison_log(outcome: ComparisonOutcome) -> str:
    """Render one UTF-8 troubleshooting record without ranked results."""

    metadata = outcome.metadata
    metadata_text = (
        f"Source: {metadata.source}\n"
        f"Target root: {metadata.target_root}\n"
        f"Model: {metadata.model_name}\n"
        f"Threshold: {metadata.threshold:.6f}\n"
        if metadata is not None
        else "Operation metadata: unavailable\n"
    )
    diagnostics = "".join(_render_diagnostic(item) for item in outcome.diagnostics)
    return (
        metadata_text
        + f"Status: {'successful' if outcome.successful else 'unsuccessful'}\n"
        + f"Target identities compared: {outcome.target_identities_compared}\n"
        + f"Candidate matches: {len(outcome.matches)}\n"
        + f"Diagnostics: {len(outcome.diagnostics)}\n"
        + diagnostics
    )


def write_comparison_artifacts(
    outcome: ComparisonOutcome,
    request: ComparisonArtifactRequest,
) -> ComparisonOutcome:
    """Write only selected artifacts and return their operation semantics."""

    if request.result_path is None and request.log_path is None:
        return outcome

    current = outcome
    if request.result_path is not None and outcome.successful:
        try:
            request.result_path.write_text(
                render_comparison_result(outcome),
                encoding="utf-8",
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
            request.log_path.write_text(
                render_comparison_log(current),
                encoding="utf-8",
            )
        except OSError as error:
            current = replace(
                current,
                diagnostics=current.diagnostics
                + (
                    Diagnostic(
                        severity="warning",
                        category="output",
                        code="log-artifact-write-failed",
                        path=request.log_path,
                        message=str(error),
                    ),
                ),
            )
    return current


def present_comparison(
    outcome: ComparisonOutcome,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Write one comparison outcome to separated CLI streams."""

    stderr.writelines(
        _render_diagnostic(diagnostic) for diagnostic in outcome.diagnostics
    )

    if not outcome.successful:
        return 1

    stdout.write(render_comparison_result(outcome))

    warning_count = sum(
        diagnostic.severity == "warning" for diagnostic in outcome.diagnostics
    )
    if warning_count:
        warning_label = "warning" if warning_count == 1 else "warnings"
        compared_count = outcome.target_identities_compared
        compared_label = (
            "target identity" if compared_count == 1 else "target identities"
        )
        match_count = len(outcome.matches)
        match_label = "candidate match" if match_count == 1 else "candidate matches"
        stderr.write(
            f"WARNING SUMMARY: {warning_count} {warning_label}; "
            f"{compared_count} {compared_label} compared; "
            f"{match_count} {match_label}.\n"
        )
    return 0
