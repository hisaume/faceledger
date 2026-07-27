"""Human-readable comparison result presentation."""

from typing import TextIO

from faceledger.comparison import ComparisonOutcome


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
        f"{rank:<6}{str(match.identity_path):<{identity_column_width}}"
        f"{match.cosine_distance:.6f}\n"
        for rank, match in enumerate(outcome.matches, start=1)
    )
    return "".join(lines)


def present_comparison(
    outcome: ComparisonOutcome,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Write one comparison outcome to separated CLI streams."""

    for diagnostic in outcome.diagnostics:
        location = f" {diagnostic.path}" if diagnostic.path is not None else ""
        stderr.write(
            f"{diagnostic.severity.upper()} "
            f"[{diagnostic.category}:{diagnostic.code}]"
            f"{location}: {diagnostic.message}\n"
        )

    if not outcome.successful:
        return 1

    metadata = outcome.metadata
    if metadata is None:
        raise ValueError("A successful comparison must have operation metadata.")
    stdout.write(
        f"Source: {metadata.source}\n"
        f"Target root: {metadata.target_root}\n"
        f"Model: {metadata.model_name}\n"
        f"Threshold: {metadata.threshold:.6f}\n\n"
    )
    stdout.write(render_matches(outcome))

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
