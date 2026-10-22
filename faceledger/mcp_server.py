from pathlib import Path
from typing import TypedDict

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from faceledger.comparison import ComparisonRequest, compare


class MatchResult(TypedDict):
    identity_path: str
    cosine_distance: float


class DiagnosticResult(TypedDict):
    severity: str
    category: str
    code: str
    path: str | None
    message: str


class CompareFolderResult(TypedDict):
    successful: bool
    complete: bool
    target_identities_compared: int
    matches: list[MatchResult]
    diagnostics: list[DiagnosticResult]


mcp = MCPServer("Faceledger")


@mcp.tool(
    title="Compare face against folder",
    annotations=ToolAnnotations(
        read_only_hint=True,
        open_world_hint=False,
    ),
)
def compare_source_to_identity_folder(
    source_image: str,
    target_folder: str,
) -> CompareFolderResult:
    """
    Compare one source face image against one target identity folder.

    Usage requirements:
    - `source_image` must be an image file.
    - `target_folder` must be a directory containing the target image.
    - The target image must be named exactly `folder.jpg`.
    - The source image must not be inside `target_folder`.

    This tool is non-recursive.
    """

    outcome = compare(
        ComparisonRequest(
            source=Path(source_image).expanduser(),
            target_root=Path(target_folder).expanduser(),
            single_target_folder=True,
        )
    )

    return {
        "successful": outcome.successful,
        "complete": outcome.complete,
        "target_identities_compared": outcome.target_identities_compared,
        "matches": [
            {
                "identity_path": match.identity_path.as_posix(),
                "cosine_distance": match.cosine_distance,
            }
            for match in outcome.matches
        ],
        "diagnostics": [
            {
                "severity": diagnostic.severity,
                "category": diagnostic.category,
                "code": diagnostic.code,
                "path": (str(diagnostic.path) if diagnostic.path is not None else None),
                "message": diagnostic.message,
            }
            for diagnostic in outcome.diagnostics
        ],
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
