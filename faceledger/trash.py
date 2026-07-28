"""Recoverable model-specific vector-cache trash."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from faceledger.comparison import Diagnostic
from faceledger.paths import application_data_root
from faceledger.vector_profiles import DEFAULT_MODEL_NAME, VECTOR_PROFILES


@dataclass(frozen=True)
class TrashRequest:
    root: Path
    model_name: str = DEFAULT_MODEL_NAME
    recursive: bool = False


@dataclass(frozen=True)
class TrashOutcome:
    action_directory: Path | None
    manifest_path: Path | None
    moved: tuple[Path, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    message: str = ""
    successful: bool = True


def _create_action_directory(trash_root: Path, action_id: str) -> Path:
    trash_root.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        name = action_id if suffix == 0 else f"{action_id}-{suffix}"
        candidate = trash_root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            json.dump(entries, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _discover_cache_entries(
    root: Path,
    suffix: str,
    recursive: bool,
    diagnostics: list[Diagnostic],
) -> tuple[Path, ...]:
    selected: list[Path] = []
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
                elif path.is_file() and path.name.endswith(suffix):
                    selected.append(path)
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
    return tuple(selected)


def trash_vector_cache(
    request: TrashRequest,
    *,
    now: Callable[[], datetime] | None = None,
) -> TrashOutcome:
    """Move exactly selected-model cache entries into recoverable trash."""

    if request.model_name not in VECTOR_PROFILES:
        return TrashOutcome(
            action_directory=None,
            manifest_path=None,
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
    root = request.root.resolve()
    root_error: OSError | None = None
    if not root.is_dir():
        root_error = NotADirectoryError(f"Maintenance root is not a directory: {root}")
    else:
        try:
            with os.scandir(root) as entries:
                next(entries, None)
        except OSError as error:
            root_error = error
    if root_error is not None:
        return TrashOutcome(
            action_directory=None,
            manifest_path=None,
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

    profile = VECTOR_PROFILES[request.model_name]
    suffix = f".{profile.cache_slug}.npy"
    diagnostics: list[Diagnostic] = []
    selected = _discover_cache_entries(
        root,
        suffix,
        recursive=request.recursive,
        diagnostics=diagnostics,
    )
    if not selected:
        return TrashOutcome(
            action_directory=None,
            manifest_path=None,
            diagnostics=tuple(diagnostics),
            message=f"No matching {profile.model_name} cache entries found",
        )

    current_time = now() if now is not None else datetime.now(UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    action_id = current_time.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    action_directory = _create_action_directory(
        application_data_root() / "trash",
        action_id,
    )
    manifest_path = action_directory / "manifest.txt"
    entries: list[dict[str, object]] = []
    destinations: list[Path] = []
    for source in selected:
        relative_destination = Path("files") / source.relative_to(root)
        entries.append(
            {
                "status": "planned",
                "original": str(source),
                "trash_relative": relative_destination.as_posix(),
                "reason": None,
            }
        )
        destinations.append(action_directory / relative_destination)
    _write_manifest(manifest_path, entries)

    moved: list[Path] = []
    for source, destination in zip(selected, destinations, strict=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        moved.append(destination)

    return TrashOutcome(
        action_directory=action_directory,
        manifest_path=manifest_path,
        moved=tuple(moved),
        diagnostics=tuple(diagnostics),
        message=f"Moved {len(moved)} {profile.model_name} cache entries to trash",
    )
