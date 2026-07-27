"""Public comparison operation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


class RecognitionAdapter(Protocol):
    """Boundary used to calculate a vector for one face image."""

    def vector_for(self, image_path: Path) -> Sequence[float]: ...


class RecognitionFailure(Exception):
    """A selected image could not produce one usable face vector."""


@dataclass(frozen=True)
class ComparisonRequest:
    target_root: Path
    source: Path | None = None
    source_folder: Path | None = None
    threshold: float = 0.30


@dataclass(frozen=True)
class CandidateMatch:
    identity_path: Path
    cosine_distance: float


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    category: str
    code: str
    path: Path | None
    message: str


@dataclass(frozen=True)
class ComparisonOutcome:
    matches: tuple[CandidateMatch, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    progress: tuple[object, ...] = ()
    successful: bool = True


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    return 1.0 - dot_product / (left_length * right_length)


def _is_recognized_folder_image(path: Path) -> bool:
    if path.name == "folder.jpg":
        return True
    suffix = path.suffix.lower()
    stem = path.stem
    if suffix == ".jpg" and len(stem) == 7:
        return stem.startswith("folder") and stem[-1].isdigit()
    return suffix in {".jpg", ".jpeg", ".png", ".webp"} and any(
        stem.lower().endswith(f".face{number}") for number in range(10)
    )


def _normalized(vector: Sequence[float]) -> tuple[float, ...]:
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector)


def _folder_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    normalized_vectors = [_normalized(vector) for vector in vectors]
    centroid = tuple(
        sum(values) / len(normalized_vectors)
        for values in zip(*normalized_vectors, strict=True)
    )
    return _normalized(centroid)


def compare(
    request: ComparisonRequest,
    recognition: RecognitionAdapter,
) -> ComparisonOutcome:
    """Compare one standalone source with the selected root identity."""

    if request.source is not None and request.source_folder is not None:
        return ComparisonOutcome(
            matches=(),
            diagnostics=(
                Diagnostic(
                    severity="error",
                    category="input",
                    code="source-selection-ambiguous",
                    path=None,
                    message="Select one source image or source folder, not both.",
                ),
            ),
            successful=False,
        )
    if request.source is None and request.source_folder is None:
        return ComparisonOutcome(
            matches=(),
            diagnostics=(
                Diagnostic(
                    severity="error",
                    category="input",
                    code="source-selection-required",
                    path=None,
                    message="Select exactly one source image or source folder.",
                ),
            ),
            successful=False,
        )

    target_root = request.target_root.resolve()
    source = request.source.resolve() if request.source is not None else None
    source_folder = (
        request.source_folder.resolve() if request.source_folder is not None else None
    )
    if source is not None and not source.is_file():
        return ComparisonOutcome(
            matches=(),
            diagnostics=(
                Diagnostic(
                    severity="error",
                    category="source",
                    code="source-image-invalid",
                    path=source,
                    message="The selected source image is not a readable file.",
                ),
            ),
            successful=False,
        )
    if source_folder is not None and (
        not source_folder.is_dir() or not (source_folder / "folder.jpg").is_file()
    ):
        return ComparisonOutcome(
            matches=(),
            diagnostics=(
                Diagnostic(
                    severity="error",
                    category="source",
                    code="source-folder-invalid",
                    path=source_folder,
                    message=(
                        "Select a single-person source folder containing exact "
                        "lowercase folder.jpg."
                    ),
                ),
            ),
            successful=False,
        )

    source_diagnostics: list[Diagnostic] = []
    if source_folder is not None:
        source_images = tuple(
            path
            for path in sorted(source_folder.iterdir())
            if path.is_file() and _is_recognized_folder_image(path)
        )
        source_vectors: list[Sequence[float]] = []
        for path in source_images:
            try:
                source_vectors.append(recognition.vector_for(path))
            except RecognitionFailure as error:
                source_diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="source",
                        code="source-folder-image-unusable",
                        path=path,
                        message=str(error),
                    )
                )
        if not source_vectors:
            source_diagnostics.append(
                Diagnostic(
                    severity="error",
                    category="source",
                    code="source-folder-unusable",
                    path=source_folder,
                    message=(
                        f"No usable source faces found in {source_folder} "
                        f"(examined: {len(source_images)}, usable: 0)."
                    ),
                )
            )
            return ComparisonOutcome(
                matches=(),
                diagnostics=tuple(source_diagnostics),
                successful=False,
            )
        source_vector = _folder_vector(source_vectors)
    else:
        try:
            source_vector = recognition.vector_for(source)
        except RecognitionFailure as error:
            return ComparisonOutcome(
                matches=(),
                diagnostics=(
                    Diagnostic(
                        severity="error",
                        category="source",
                        code="source-image-unusable",
                        path=source,
                        message=str(error),
                    ),
                ),
                successful=False,
            )

    target_image = target_root / "folder.jpg"
    distance = _cosine_distance(
        source_vector,
        recognition.vector_for(target_image),
    )
    matches = (
        (CandidateMatch(identity_path=Path("."), cosine_distance=distance),)
        if distance <= request.threshold
        else ()
    )
    return ComparisonOutcome(
        matches=matches,
        diagnostics=tuple(source_diagnostics),
    )
