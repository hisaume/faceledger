"""Explicit vector-cache maintenance operations."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from faceledger.comparison import (
    AssetAcquisitionFailure,
    Diagnostic,
    InvalidCacheEntry,
    RecognitionAdapter,
    RecognitionFailure,
    _cache_path,
    _folder_vector,
    _is_animated_webp,
    _is_numbered_face_image,
    _is_recognized_folder_image,
    _load_cached_vector,
)
from faceledger.vector_profiles import (
    DEFAULT_MODEL_NAME,
    VECTOR_PROFILES,
    VectorProfile,
)


@dataclass(frozen=True)
class CacheBuildRequest:
    root: Path
    model_name: str = DEFAULT_MODEL_NAME
    recursive: bool = False


@dataclass(frozen=True)
class CacheBuildOutcome:
    created: tuple[Path, ...]
    retained: tuple[Path, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    successful: bool = True


def _validated_vector(
    vector: Sequence[float],
    profile: VectorProfile,
) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError, OverflowError) as error:
        raise RecognitionFailure("Recognition returned a non-numeric vector.") from error
    if len(values) != profile.expected_dimensions:
        raise RecognitionFailure(
            f"Expected {profile.expected_dimensions} vector dimensions; "
            f"received {len(values)}."
        )
    if not all(math.isfinite(value) for value in values):
        raise RecognitionFailure("Recognition returned a non-finite vector.")
    return values


def _persist_vector(cache_path: Path, vector: Sequence[float]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            np.save(temporary_file, np.asarray(vector, dtype=float), allow_pickle=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, cache_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _maintenance_folder_views(
    root: Path,
    recursive: bool,
    diagnostics: list[Diagnostic],
) -> Iterator[tuple[Path, tuple[Path, ...]]]:
    pending = [root]
    while pending:
        folder = pending.pop()
        try:
            entries = tuple(sorted(folder.iterdir()))
        except OSError as error:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    category="maintenance",
                    code="maintenance-descendant-unavailable",
                    path=folder,
                    message=str(error),
                )
            )
            continue

        files: list[Path] = []
        children: list[Path] = []
        for path in entries:
            try:
                if path.is_symlink():
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            category="maintenance",
                            code="maintenance-symlink-skipped",
                            path=path,
                            message="Discovered maintenance symlinks are not followed.",
                        )
                    )
                elif path.is_dir():
                    children.append(path)
                elif path.is_file():
                    files.append(path)
            except OSError as error:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="maintenance",
                        code="maintenance-item-unavailable",
                        path=path,
                        message=str(error),
                    )
                )
        if recursive:
            pending.extend(reversed(children))
        yield folder, tuple(files)


def build_vector_cache(
    request: CacheBuildRequest,
    recognition: RecognitionAdapter | None = None,
) -> CacheBuildOutcome:
    """Create missing selected-model cache entries in one maintenance root."""

    if request.model_name not in VECTOR_PROFILES:
        return CacheBuildOutcome(
            created=(),
            diagnostics=(
                Diagnostic(
                    severity="error",
                    category="input",
                    code="recognition-model-unsupported",
                    path=None,
                    message=f"Unsupported recognition model: {request.model_name}.",
                ),
            ),
            successful=False,
        )
    profile = VECTOR_PROFILES[request.model_name]
    root = request.root.resolve()
    root_error: OSError | None = None
    if not root.is_dir():
        root_error = NotADirectoryError(
            f"Maintenance root is not a directory: {root}"
        )
    else:
        try:
            with os.scandir(root) as entries:
                next(entries, None)
        except OSError as error:
            root_error = error
    if root_error is not None:
        return CacheBuildOutcome(
            created=(),
            diagnostics=(
                Diagnostic(
                    severity="error",
                    category="maintenance",
                    code="maintenance-root-invalid",
                    path=root,
                    message=str(root_error),
                ),
            ),
            successful=False,
        )

    diagnostics: list[Diagnostic] = []
    created: list[Path] = []
    retained: list[Path] = []
    if recognition is None:
        from faceledger.deepface_adapter import DeepFaceRecognition

        def announce_missing_asset(path: Path) -> None:
            diagnostics.append(
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

    def asset_failure_outcome(error: AssetAcquisitionFailure) -> CacheBuildOutcome:
        diagnostics.append(
            Diagnostic(
                severity="error",
                category="dependency",
                code="model-assets-unavailable",
                path=None,
                message=str(error),
            )
        )
        return CacheBuildOutcome(
            created=tuple(created),
            retained=tuple(retained),
            diagnostics=tuple(diagnostics),
            successful=False,
        )

    for _folder, files in _maintenance_folder_views(
        root,
        recursive=request.recursive,
        diagnostics=diagnostics,
    ):
        anchor = next((path for path in files if path.name == "folder.jpg"), None)
        if anchor is not None:
            cache_path = _cache_path(anchor, profile)
            if cache_path.is_symlink():
                continue
            try:
                cached_vector = _load_cached_vector(anchor, profile)
            except InvalidCacheEntry as error:
                cached_vector = None
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="maintenance",
                        code="cache-entry-invalid",
                        path=cache_path,
                        message=str(error),
                    )
                )
            if cached_vector is not None:
                retained.append(cache_path)
                continue

            image_vectors: list[Sequence[float]] = []
            for image_path in files:
                if not _is_recognized_folder_image(image_path):
                    continue
                if _is_animated_webp(image_path):
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            category="maintenance",
                            code="animated-webp-unsupported",
                            path=image_path,
                            message="Animated WebP face images are not supported.",
                        )
                    )
                    continue
                try:
                    image_vectors.append(
                        _validated_vector(
                            recognition.vector_for(image_path, profile),
                            profile,
                        )
                    )
                except AssetAcquisitionFailure as error:
                    return asset_failure_outcome(error)
                except (RecognitionFailure, OSError) as error:
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            category="maintenance",
                            code="cache-build-image-unusable",
                            path=image_path,
                            message=str(error),
                        )
                    )
            if not image_vectors:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="maintenance",
                        code="cache-build-identity-unusable",
                        path=anchor,
                        message="No usable face images remain for this folder identity.",
                    )
                )
                continue
            try:
                _persist_vector(cache_path, _folder_vector(image_vectors))
            except OSError as error:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="maintenance",
                        code="cache-build-write-failed",
                        path=cache_path,
                        message=str(error),
                    )
                )
            else:
                created.append(cache_path)
            continue

        for image_path in files:
            if not _is_numbered_face_image(image_path):
                continue
            if _is_animated_webp(image_path):
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="maintenance",
                        code="animated-webp-unsupported",
                        path=image_path,
                        message="Animated WebP face images are not supported.",
                    )
                )
                continue
            cache_path = _cache_path(image_path, profile)
            if cache_path.is_symlink():
                continue
            try:
                cached_vector = _load_cached_vector(image_path, profile)
            except InvalidCacheEntry as error:
                cached_vector = None
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="maintenance",
                        code="cache-entry-invalid",
                        path=cache_path,
                        message=str(error),
                    )
                )
            if cached_vector is not None:
                retained.append(cache_path)
                continue
            try:
                vector = _validated_vector(
                    recognition.vector_for(image_path, profile),
                    profile,
                )
            except AssetAcquisitionFailure as error:
                return asset_failure_outcome(error)
            except (RecognitionFailure, OSError) as error:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="maintenance",
                        code="cache-build-image-unusable",
                        path=image_path,
                        message=str(error),
                    )
                )
                continue
            try:
                _persist_vector(cache_path, vector)
            except OSError as error:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="maintenance",
                        code="cache-build-write-failed",
                        path=cache_path,
                        message=str(error),
                    )
                )
                continue
            created.append(cache_path)

    return CacheBuildOutcome(
        created=tuple(created),
        retained=tuple(retained),
        diagnostics=tuple(diagnostics),
    )
