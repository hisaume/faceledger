"""Public comparison operation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


class RecognitionAdapter(Protocol):
    """Boundary used to calculate a vector for one face image."""

    def vector_for(self, image_path: Path) -> Sequence[float]: ...


@dataclass(frozen=True)
class ComparisonRequest:
    source: Path
    target_root: Path
    threshold: float = 0.30


@dataclass(frozen=True)
class CandidateMatch:
    identity_path: Path
    cosine_distance: float


@dataclass(frozen=True)
class ComparisonOutcome:
    matches: tuple[CandidateMatch, ...]
    diagnostics: tuple[object, ...] = ()
    progress: tuple[object, ...] = ()


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    return 1.0 - dot_product / (left_length * right_length)


def compare(
    request: ComparisonRequest,
    recognition: RecognitionAdapter,
) -> ComparisonOutcome:
    """Compare one standalone source with the selected root identity."""

    target_image = request.target_root / "folder.jpg"
    distance = _cosine_distance(
        recognition.vector_for(request.source),
        recognition.vector_for(target_image),
    )
    matches = (
        (CandidateMatch(identity_path=Path("."), cosine_distance=distance),)
        if distance <= request.threshold
        else ()
    )
    return ComparisonOutcome(matches=matches)
