"""Public comparison operation."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from faceledger.vector_profiles import (
    DEFAULT_MODEL_NAME,
    VECTOR_PROFILES,
    VectorProfile,
)

type Embedding = tuple[float, ...]


class RecognitionAdapter(Protocol):
    """Boundary used to calculate a vector for one face image."""

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> Sequence[float]: ...


class RecognitionFailure(Exception):
    """A selected image could not produce one usable face vector."""


class AssetAcquisitionFailure(Exception):
    """DeepFace could not acquire dependency-owned model assets."""


class InvalidCacheEntry(Exception):
    """A model-qualified vector-cache entry is structurally incompatible."""


@dataclass(frozen=True)
class ComparisonRequest:
    target_root: Path
    source: Path | None = None
    source_folder: Path | None = None
    model_name: str = DEFAULT_MODEL_NAME
    threshold: float | None = None
    single_target_folder: bool = False
    reuse_cache: bool = True


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


class _DiagnosticCollector(list[Diagnostic]):
    """Retain diagnostics and stream each one as it arises."""

    def __init__(
        self,
        on_diagnostic: Callable[[Diagnostic], None] | None,
    ) -> None:
        """Configure the optional presentation callback."""

        super().__init__()
        self._on_diagnostic = on_diagnostic

    def append(self, diagnostic: Diagnostic) -> None:
        """Retain a diagnostic before presenting it to the callback."""

        super().append(diagnostic)
        if self._on_diagnostic is not None:
            self._on_diagnostic(diagnostic)


@dataclass(frozen=True)
class ComparisonMetadata:
    source: Path
    target_root: Path
    model_name: str
    threshold: float


@dataclass(frozen=True)
class ProgressNotification:
    """Presentation-neutral notice for one completed operation item."""

    category: str
    completed_items: int
    path: Path
    message: str


@dataclass(frozen=True)
class ComparisonOutcome:
    matches: tuple[CandidateMatch, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    progress: tuple[ProgressNotification, ...] = ()
    successful: bool = True
    complete: bool = True
    target_identities_compared: int = 0
    metadata: ComparisonMetadata | None = None


def _cosine_distance(left: Embedding, right: Embedding) -> float:
    """Calculate cosine distance between two equal-length vectors."""

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    return 1.0 - dot_product / (left_length * right_length)


def _is_recognized_folder_image(path: Path) -> bool:
    """Identify images that contribute to a single-person folder vector."""

    if path.name == "folder.jpg":
        return True
    suffix = path.suffix.lower()
    stem = path.stem
    if suffix == ".jpg" and len(stem) == 7:
        return stem.startswith("folder") and stem[-1].isdigit()
    return _is_numbered_face_image(path)


def _is_numbered_face_image(path: Path) -> bool:
    """Identify one-digit numbered face images using v1 naming rules."""

    suffix = path.suffix.lower()
    stem = path.stem
    return suffix in {".jpg", ".jpeg", ".png", ".webp"} and any(
        stem.lower().endswith(f".face{number}") for number in range(10)
    )


def _is_animated_webp(path: Path) -> bool:
    """Detect animated WebP without classifying unreadable files as animated."""

    if path.suffix.lower() != ".webp":
        return False
    try:
        with Image.open(path) as image:
            return bool(getattr(image, "is_animated", False)) or (
                getattr(image, "n_frames", 1) > 1
            )
    except OSError:
        return False


def _normalized(vector: Embedding) -> Embedding:
    """Return a unit-length copy of a vector."""

    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector)


def _folder_vector(vectors: Sequence[Embedding]) -> Embedding:
    """Build the normalized equal-weight centroid for one folder identity."""

    normalized_vectors = [_normalized(vector) for vector in vectors]
    centroid = tuple(
        sum(values) / len(normalized_vectors)
        for values in zip(*normalized_vectors, strict=True)
    )
    return _normalized(centroid)


def _cache_path(image_path: Path, profile: VectorProfile) -> Path:
    return image_path.with_name(f"{image_path.name}.{profile.cache_slug}.npy")


def _load_cached_vector(
    image_path: Path,
    profile: VectorProfile,
) -> Embedding | None:
    """Load a cache vector only when it matches the selected profile."""

    cache_path = _cache_path(image_path, profile)
    if cache_path.is_symlink() or not cache_path.is_file():
        return None
    try:
        vector = np.load(cache_path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise InvalidCacheEntry(
            "Cache entry could not be loaded as a non-pickled NPY vector."
        ) from error
    if not isinstance(vector, np.ndarray):
        vector.close()
        raise InvalidCacheEntry("Cache entry is not a single NPY vector.")
    if (
        vector.ndim != 1
        or vector.shape[0] != profile.expected_dimensions
        or not np.issubdtype(vector.dtype, np.number)
        or np.issubdtype(vector.dtype, np.complexfloating)
    ):
        raise InvalidCacheEntry(
            f"Expected a {profile.expected_dimensions}-dimension numeric vector."
        )
    return tuple(float(value) for value in vector)


def _reuse_cached_vector(
    image_path: Path,
    profile: VectorProfile,
    diagnostics: list[Diagnostic],
    category: str,
) -> Embedding | None:
    """Reuse a compatible cache or warn and fall back to calculation."""

    try:
        return _load_cached_vector(image_path, profile)
    except InvalidCacheEntry as error:
        diagnostics.append(
            Diagnostic(
                severity="warning",
                category=category,
                code=f"{category}-cache-invalid",
                path=_cache_path(image_path, profile),
                message=str(error),
            )
        )
        return None


def _warn_target_unavailable(
    diagnostics: list[Diagnostic],
    path: Path,
    error: OSError,
) -> None:
    diagnostics.append(
        Diagnostic(
            severity="warning",
            category="target",
            code="target-descendant-unavailable",
            path=path,
            message=str(error),
        )
    )


def _target_folder_views(
    target_root: Path,
    recursive: bool,
    diagnostics: list[Diagnostic],
) -> Iterator[tuple[Path, tuple[Path, ...]]]:
    """Yield target-folder file views without following discovered symlinks."""

    pending = [target_root]
    while pending:
        target_folder = pending.pop()
        try:
            target_entries = tuple(sorted(target_folder.iterdir()))
        except OSError as error:
            _warn_target_unavailable(diagnostics, target_folder, error)
            continue

        regular_files: list[Path] = []
        child_directories: list[Path] = []
        for path in target_entries:
            try:
                if path.is_symlink():
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            category="target",
                            code="target-symlink-skipped",
                            path=path,
                            message="Discovered target symlinks are not followed.",
                        )
                    )
                elif path.is_dir():
                    child_directories.append(path)
                elif path.is_file():
                    regular_files.append(path)
                elif not path.exists():
                    _warn_target_unavailable(
                        diagnostics,
                        path,
                        FileNotFoundError(f"Discovered target disappeared: {path}"),
                    )
            except OSError as error:
                _warn_target_unavailable(diagnostics, path, error)

        if recursive:
            pending.extend(reversed(child_directories))
        yield target_folder, tuple(regular_files)


def compare(
    request: ComparisonRequest,
    recognition: RecognitionAdapter | None = None,
    *,
    on_diagnostic: Callable[[Diagnostic], None] | None = None,
    on_progress: Callable[[ProgressNotification], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> ComparisonOutcome:
    """Compare one standalone source with the selected root identity."""

    diagnostics = _DiagnosticCollector(on_diagnostic)

    def emit_diagnostic(diagnostic: Diagnostic) -> None:
        """Retain and stream one diagnostic notification."""

        diagnostics.append(diagnostic)

    if request.model_name not in VECTOR_PROFILES:
        emit_diagnostic(
            Diagnostic(
                severity="error",
                category="input",
                code="recognition-model-unsupported",
                path=None,
                message=f"Unsupported recognition model: {request.model_name}.",
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            successful=False,
        )
    profile = VECTOR_PROFILES[request.model_name]
    if request.threshold is not None and (
        not math.isfinite(request.threshold) or not 0.0 <= request.threshold <= 2.0
    ):
        emit_diagnostic(
            Diagnostic(
                severity="error",
                category="input",
                code="match-threshold-invalid",
                path=None,
                message="The match threshold must be finite and within [0, 2].",
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            successful=False,
        )
    active_threshold = (
        profile.cosine_threshold if request.threshold is None else request.threshold
    )

    if request.source is not None and request.source_folder is not None:
        emit_diagnostic(
            Diagnostic(
                severity="error",
                category="input",
                code="source-selection-ambiguous",
                path=None,
                message="Select one source image or source folder, not both.",
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            successful=False,
        )
    if request.source is None and request.source_folder is None:
        emit_diagnostic(
            Diagnostic(
                severity="error",
                category="input",
                code="source-selection-required",
                path=None,
                message="Select exactly one source image or source folder.",
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            successful=False,
        )

    target_root = request.target_root.resolve()
    source = request.source.resolve() if request.source is not None else None
    source_folder = (
        request.source_folder.resolve() if request.source_folder is not None else None
    )
    target_root_error: OSError | None = None
    if not target_root.is_dir():
        target_root_error = NotADirectoryError(
            f"Target root is not a directory: {target_root}"
        )
    else:
        try:
            with os.scandir(target_root) as entries:
                next(entries, None)
        except OSError as error:
            target_root_error = error
    if target_root_error is not None:
        emit_diagnostic(
            Diagnostic(
                severity="error",
                category="target",
                code="target-root-invalid",
                path=target_root,
                message=str(target_root_error),
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            successful=False,
        )
    if source is not None and not source.is_file():
        emit_diagnostic(
            Diagnostic(
                severity="error",
                category="source",
                code="source-image-invalid",
                path=source,
                message="The selected source image is not a readable file.",
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            successful=False,
        )
    if source_folder is not None and (
        not source_folder.is_dir() or not (source_folder / "folder.jpg").is_file()
    ):
        emit_diagnostic(
            Diagnostic(
                severity="error",
                category="source",
                code="source-folder-invalid",
                path=source_folder,
                message=(
                    "Select a single-person source folder containing exact "
                    "lowercase folder.jpg."
                ),
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            successful=False,
        )

    progress: list[ProgressNotification] = []
    if recognition is None:
        from faceledger.deepface_adapter import DeepFaceRecognition

        def announce_missing_asset(path: Path) -> None:
            emit_diagnostic(
                Diagnostic(
                    severity="info",
                    category="dependency",
                    code="model-asset-acquisition",
                    path=path,
                    message=(
                        f"DeepFace will acquire missing dependency asset {path.name}."
                    ),
                )
            )

        recognition = DeepFaceRecognition(announce_missing_asset)

    def calculate_vector(path: Path, category: str) -> Embedding:
        """Calculate an uncached vector and emit its completion progress."""

        try:
            return tuple(recognition.vector_for(path, profile))
        finally:
            notification = ProgressNotification(
                category=category,
                completed_items=len(progress) + 1,
                path=path,
                message=f"Completed uncached {category} image: {path}",
            )
            progress.append(notification)
            if on_progress is not None:
                on_progress(notification)

    def asset_failure_outcome(error: AssetAcquisitionFailure) -> ComparisonOutcome:
        """Translate asset failure into the comparison operation contract."""

        diagnostics.append(
            Diagnostic(
                severity="error",
                category="dependency",
                code="model-assets-unavailable",
                path=None,
                message=str(error),
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            progress=tuple(progress),
            successful=False,
            metadata=metadata,
        )

    resolved_source = source if source is not None else source_folder
    if resolved_source is None:
        raise AssertionError("Validated comparison has no resolved source.")
    metadata = ComparisonMetadata(
        source=resolved_source,
        target_root=target_root,
        model_name=profile.model_name,
        threshold=active_threshold,
    )

    def is_cancelled() -> bool:
        return cancellation_requested is not None and cancellation_requested()

    def cancelled_outcome(compared: int = 0) -> ComparisonOutcome:
        """Discard candidates and report an incomplete comparison."""

        emit_diagnostic(
            Diagnostic(
                severity="info",
                category="operation",
                code="comparison-cancelled",
                path=None,
                message="Comparison cancelled; the operation is incomplete.",
            )
        )
        return ComparisonOutcome(
            matches=(),
            diagnostics=tuple(diagnostics),
            progress=tuple(progress),
            successful=False,
            complete=False,
            target_identities_compared=compared,
            metadata=metadata,
        )

    if is_cancelled():
        return cancelled_outcome()

    source_vector: Embedding
    if source_folder is not None:
        cached_source_vector = (
            _reuse_cached_vector(
                source_folder / "folder.jpg",
                profile,
                diagnostics,
                category="source",
            )
            if request.reuse_cache
            else None
        )
        if cached_source_vector is None:
            source_images = tuple(
                path
                for path in sorted(source_folder.iterdir())
                if path.is_file() and _is_recognized_folder_image(path)
            )
            source_vectors: list[Embedding] = []
            for path in source_images:
                if is_cancelled():
                    return cancelled_outcome()
                try:
                    source_vectors.append(calculate_vector(path, "source"))
                except AssetAcquisitionFailure as error:
                    return asset_failure_outcome(error)
                except RecognitionFailure as error:
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            category="source",
                            code="source-folder-image-unusable",
                            path=path,
                            message=str(error),
                        )
                    )
            if not source_vectors:
                diagnostics.append(
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
                    diagnostics=tuple(diagnostics),
                    progress=tuple(progress),
                    successful=False,
                    metadata=metadata,
                )
            source_vector = _folder_vector(source_vectors)
        else:
            source_vector = cached_source_vector
    else:
        assert source is not None
        try:
            source_vector = calculate_vector(source, "source")
        except AssetAcquisitionFailure as error:
            return asset_failure_outcome(error)
        except RecognitionFailure as error:
            emit_diagnostic(
                Diagnostic(
                    severity="error",
                    category="source",
                    code="source-image-unusable",
                    path=source,
                    message=str(error),
                )
            )
            return ComparisonOutcome(
                matches=(),
                diagnostics=tuple(diagnostics),
                progress=tuple(progress),
                successful=False,
                metadata=metadata,
            )

    discovered_identities: list[tuple[Path, Embedding]] = []
    for target_folder, regular_entries in _target_folder_views(
        target_root,
        recursive=not request.single_target_folder,
        diagnostics=diagnostics,
    ):
        if is_cancelled():
            return cancelled_outcome(len(discovered_identities))
        target_image = next(
            (path for path in regular_entries if path.name == "folder.jpg"),
            None,
        )
        if target_image is not None:
            if source_folder == target_folder or (
                source is not None and source.parent == target_folder
            ):
                continue
            cached_target_vector = (
                _reuse_cached_vector(
                    target_image,
                    profile,
                    diagnostics,
                    category="target",
                )
                if request.reuse_cache
                else None
            )
            if cached_target_vector is not None:
                discovered_identities.append(
                    (target_folder.relative_to(target_root), cached_target_vector)
                )
                continue
            target_images = tuple(
                path for path in regular_entries if _is_recognized_folder_image(path)
            )
            target_vectors: list[Embedding] = []
            for path in target_images:
                if is_cancelled():
                    return cancelled_outcome(len(discovered_identities))
                try:
                    target_vectors.append(calculate_vector(path, "target"))
                except AssetAcquisitionFailure as error:
                    return asset_failure_outcome(error)
                except (RecognitionFailure, OSError) as error:
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            category="target",
                            code="target-folder-image-unusable",
                            path=path,
                            message=str(error),
                        )
                    )
            if target_vectors:
                discovered_identities.append(
                    (
                        target_folder.relative_to(target_root),
                        _folder_vector(target_vectors),
                    )
                )
            continue

        for path in regular_entries:
            if is_cancelled():
                return cancelled_outcome(len(discovered_identities))
            if not _is_numbered_face_image(path):
                continue
            if source is not None and path == source:
                continue
            if _is_animated_webp(path):
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="target",
                        code="animated-webp-unsupported",
                        path=path,
                        message="Animated WebP target images are not supported.",
                    )
                )
                continue
            try:
                cached_target_vector = (
                    _reuse_cached_vector(
                        path,
                        profile,
                        diagnostics,
                        category="target",
                    )
                    if request.reuse_cache
                    else None
                )
                if cached_target_vector is None:
                    target_vector = calculate_vector(path, "target")
                else:
                    target_vector = cached_target_vector
            except AssetAcquisitionFailure as error:
                return asset_failure_outcome(error)
            except (RecognitionFailure, OSError) as error:
                emit_diagnostic(
                    Diagnostic(
                        severity="warning",
                        category="target",
                        code="target-face-unusable",
                        path=path,
                        message=str(error),
                    )
                )
            else:
                discovered_identities.append(
                    (path.relative_to(target_root), target_vector)
                )
    if is_cancelled():
        return cancelled_outcome(len(discovered_identities))
    target_identities = tuple(discovered_identities)
    matches = tuple(
        sorted(
            (
                CandidateMatch(identity_path=identity_path, cosine_distance=distance)
                for identity_path, target_vector in target_identities
                if (distance := _cosine_distance(source_vector, target_vector))
                <= active_threshold
            ),
            key=lambda match: (
                match.cosine_distance,
                match.identity_path.as_posix(),
            ),
        )
    )
    return ComparisonOutcome(
        matches=matches,
        diagnostics=tuple(diagnostics),
        progress=tuple(progress),
        target_identities_compared=len(target_identities),
        metadata=metadata,
    )
