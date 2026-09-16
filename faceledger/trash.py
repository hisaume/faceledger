"""Recoverable model-specific vector-cache trash."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from faceledger.comparison import Diagnostic, ProgressNotification, _DiagnosticCollector
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
    progress: tuple[ProgressNotification, ...] = ()
    complete: bool = True


def _create_action_directory(trash_root: Path, action_id: str) -> Path:
    """Create a collision-safe recovery directory for one trash action."""

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
    """Atomically persist the complete current trash manifest."""

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


def _files_match(source: Path, copied: Path) -> bool:
    """Verify a copied cache entry matches its source byte for byte."""

    with source.open("rb") as source_file, copied.open("rb") as copied_file:
        while True:
            source_chunk = source_file.read(1024 * 1024)
            copied_chunk = copied_file.read(1024 * 1024)
            if source_chunk != copied_chunk:
                return False
            if not source_chunk:
                return True


def _copy_across_filesystems(source: Path, destination: Path) -> None:
    """Copy and verify a cache before removing its cross-filesystem source."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    destination_installed = False
    try:
        with (
            source.open("rb") as source_file,
            os.fdopen(
                descriptor,
                "wb",
            ) as copied_file,
        ):
            shutil.copyfileobj(source_file, copied_file)
            copied_file.flush()
            os.fsync(copied_file.fileno())
        shutil.copystat(source, temporary_path)
        if not _files_match(source, temporary_path):
            raise OSError("Copied cache verification failed.")
        os.replace(temporary_path, destination)
        destination_installed = True
        source.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        if destination_installed:
            destination.unlink(missing_ok=True)
        raise


def _discover_cache_entries(
    root: Path,
    suffix: str,
    recursive: bool,
    diagnostics: list[Diagnostic],
) -> tuple[Path, ...]:
    """Discover exact-model caches without following filesystem symlinks."""

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
    on_diagnostic: Callable[[Diagnostic], None] | None = None,
    on_progress: Callable[[ProgressNotification], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> TrashOutcome:
    """Move exactly selected-model cache entries into recoverable trash."""

    diagnostics = _DiagnosticCollector(on_diagnostic)
    if request.model_name not in VECTOR_PROFILES:
        diagnostics.append(
            Diagnostic(
                severity="error",
                category="input",
                code="recognition-model-unsupported",
                path=None,
                message=f"Unsupported recognition model: {request.model_name}.",
            )
        )
        return TrashOutcome(
            action_directory=None,
            manifest_path=None,
            diagnostics=tuple(diagnostics),
            successful=False,
        )
    root = request.root.resolve()
    root_error: OSError | None = None
    if not root.is_dir():
        root_error = NotADirectoryError(f"Maintenance root is not a directory: {root}")
    else:
        try:
            with os.scandir(root) as directory_entries:
                next(directory_entries, None)
        except OSError as error:
            root_error = error
    if root_error is not None:
        diagnostics.append(
            Diagnostic(
                severity="error",
                category="maintenance",
                code="maintenance-root-invalid",
                path=root,
                message=str(root_error),
            )
        )
        return TrashOutcome(
            action_directory=None,
            manifest_path=None,
            diagnostics=tuple(diagnostics),
            successful=False,
        )

    profile = VECTOR_PROFILES[request.model_name]
    suffix = f".{profile.cache_slug}.npy"
    progress: list[ProgressNotification] = []
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

    def is_cancelled() -> bool:
        return cancellation_requested is not None and cancellation_requested()

    def emit_progress(path: Path) -> None:
        """Publish completion only after the manifest records an attempt."""

        notification = ProgressNotification(
            category="trash",
            completed_items=len(progress) + 1,
            path=path,
            message=f"Completed trash entry: {path}",
        )
        progress.append(notification)
        if on_progress is not None:
            on_progress(notification)

    def cancelled_outcome() -> TrashOutcome:
        """Report incomplete trash work while preserving its durable manifest."""

        diagnostics.append(
            Diagnostic(
                severity="info",
                category="operation",
                code="trash-cancelled",
                path=None,
                message="Trash operation cancelled; the operation is incomplete.",
            )
        )
        return TrashOutcome(
            action_directory=action_directory,
            manifest_path=manifest_path,
            moved=tuple(moved),
            diagnostics=tuple(diagnostics),
            message=f"Moved {len(moved)} {profile.model_name} cache entries to trash",
            successful=False,
            progress=tuple(progress),
            complete=False,
        )

    if is_cancelled():
        return cancelled_outcome()

    for index, (source, destination) in enumerate(
        zip(selected, destinations, strict=True)
    ):
        if is_cancelled():
            return cancelled_outcome()
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        except OSError as error:
            if error.errno == errno.EXDEV:
                try:
                    _copy_across_filesystems(source, destination)
                except OSError as copy_error:
                    error = copy_error
                else:
                    moved.append(destination)
                    entries[index]["status"] = "moved"
                    _write_manifest(manifest_path, entries)
                    emit_progress(source)
                    continue
            entries[index]["status"] = "failed"
            entries[index]["reason"] = str(error)
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    category="maintenance",
                    code="trash-entry-move-failed",
                    path=source,
                    message=str(error),
                )
            )
            _write_manifest(manifest_path, entries)
            emit_progress(source)
            continue
        moved.append(destination)
        entries[index]["status"] = "moved"
        _write_manifest(manifest_path, entries)
        emit_progress(source)

    if is_cancelled():
        return cancelled_outcome()

    return TrashOutcome(
        action_directory=action_directory,
        manifest_path=manifest_path,
        moved=tuple(moved),
        diagnostics=tuple(diagnostics),
        message=f"Moved {len(moved)} {profile.model_name} cache entries to trash",
        progress=tuple(progress),
    )
