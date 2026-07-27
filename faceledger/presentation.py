"""Human-readable comparison result presentation."""

from faceledger.comparison import ComparisonOutcome


def render_matches(outcome: ComparisonOutcome) -> str:
    """Render candidate matches for standard output."""

    if not outcome.successful:
        return ""
    if not outcome.matches:
        return "No matches found\n"
    lines = ["Rank  Identity  Cosine distance\n"]
    lines.extend(
        f"{rank:<6}{str(match.identity_path):<10}{match.cosine_distance:.6f}\n"
        for rank, match in enumerate(outcome.matches, start=1)
    )
    return "".join(lines)
