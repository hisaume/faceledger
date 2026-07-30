import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from faceledger.comparison import ProgressNotification
from faceledger.maintenance import (
    CacheBuildRequest,
    build_vector_cache,
    rebuild_vector_cache,
)
from faceledger.trash import TrashRequest, trash_vector_cache


def unit_vector(index: int) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(512))


class RecordingRecognition:
    def __init__(self, vectors: dict[Path, tuple[float, ...]]) -> None:
        self._vectors = vectors
        self.calls: list[Path] = []

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        self.calls.append(image_path)
        return self._vectors[image_path]


class MaintenanceCancellationTests(unittest.TestCase):
    def test_cache_build_stops_before_the_next_item_and_keeps_completed_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_face = root / "Alice.face0.jpg"
            first_face.write_bytes(b"first")
            unprocessed_face = root / "Bob.face1.jpg"
            unprocessed_face.write_bytes(b"second")
            recognition = RecordingRecognition(
                {
                    first_face: unit_vector(0),
                    unprocessed_face: unit_vector(1),
                }
            )
            observed: list[ProgressNotification] = []
            cancellation = {"requested": False}

            def observe(notification: ProgressNotification) -> None:
                observed.append(notification)
                cancellation["requested"] = True

            outcome = build_vector_cache(
                CacheBuildRequest(root=root),
                recognition,
                on_progress=observe,
                cancellation_requested=lambda: cancellation["requested"],
            )

            first_cache = root / "Alice.face0.jpg.facenet512.npy"
            unprocessed_cache = root / "Bob.face1.jpg.facenet512.npy"
            self.assertFalse(outcome.successful)
            self.assertFalse(outcome.complete)
            self.assertEqual(outcome.created, (first_cache,))
            self.assertEqual(outcome.progress, tuple(observed))
            self.assertEqual(
                [(item.category, item.completed_items, item.path) for item in observed],
                [("cache-build", 1, first_face)],
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["cache-build-cancelled"],
            )
            self.assertEqual(recognition.calls, [first_face])
            np.testing.assert_array_equal(np.load(first_cache), unit_vector(0))
            self.assertFalse(unprocessed_cache.exists())

    def test_cache_rebuild_stops_before_the_next_item_and_keeps_completed_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_face = root / "Alice.face0.jpg"
            first_face.write_bytes(b"first")
            first_cache = root / "Alice.face0.jpg.facenet512.npy"
            np.save(first_cache, np.asarray(unit_vector(2)))
            unprocessed_face = root / "Bob.face1.jpg"
            unprocessed_face.write_bytes(b"second")
            unprocessed_cache = root / "Bob.face1.jpg.facenet512.npy"
            np.save(unprocessed_cache, np.asarray(unit_vector(3)))
            unprocessed_bytes = unprocessed_cache.read_bytes()
            recognition = RecordingRecognition(
                {
                    first_face: unit_vector(0),
                    unprocessed_face: unit_vector(1),
                }
            )
            observed: list[ProgressNotification] = []
            cancellation = {"requested": False}

            def observe(notification: ProgressNotification) -> None:
                observed.append(notification)
                cancellation["requested"] = True

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root),
                recognition,
                on_progress=observe,
                cancellation_requested=lambda: cancellation["requested"],
            )

            self.assertFalse(outcome.successful)
            self.assertFalse(outcome.complete)
            self.assertEqual(outcome.rebuilt, (first_cache,))
            self.assertEqual(outcome.progress, tuple(observed))
            self.assertEqual(
                [(item.category, item.completed_items, item.path) for item in observed],
                [("cache-rebuild", 1, first_face)],
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["cache-rebuild-cancelled"],
            )
            self.assertEqual(recognition.calls, [first_face])
            np.testing.assert_array_equal(np.load(first_cache), unit_vector(0))
            self.assertEqual(unprocessed_cache.read_bytes(), unprocessed_bytes)

    def test_trash_cancellation_keeps_moved_and_planned_manifest_states(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            first_cache = root / "Alice.face0.jpg.facenet512.npy"
            first_cache.write_bytes(b"first")
            unprocessed_cache = root / "Bob.face1.jpg.facenet512.npy"
            unprocessed_cache.write_bytes(b"second")
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 30, 12, 0, 0, tzinfo=UTC)
            observed: list[ProgressNotification] = []
            cancellation = {"requested": False}

            def observe(notification: ProgressNotification) -> None:
                observed.append(notification)
                cancellation["requested"] = True

            with patch.dict(
                os.environ,
                {"XDG_DATA_HOME": str(xdg_data_home)},
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root),
                    now=lambda: fixed_time,
                    on_progress=observe,
                    cancellation_requested=lambda: cancellation["requested"],
                )

            action = xdg_data_home / "faceledger" / "trash" / "20260730T120000.000000Z"
            first_destination = action / "files" / first_cache.name
            manifest_path = action / "manifest.txt"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(outcome.successful)
            self.assertFalse(outcome.complete)
            self.assertEqual(outcome.moved, (first_destination,))
            self.assertEqual(outcome.progress, tuple(observed))
            self.assertEqual(
                [(item.category, item.completed_items, item.path) for item in observed],
                [("trash", 1, first_cache)],
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["trash-cancelled"],
            )
            self.assertFalse(first_cache.exists())
            self.assertEqual(first_destination.read_bytes(), b"first")
            self.assertEqual(unprocessed_cache.read_bytes(), b"second")
            self.assertEqual(
                [entry["status"] for entry in manifest],
                ["moved", "planned"],
            )


if __name__ == "__main__":
    unittest.main()
