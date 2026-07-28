import io
import tempfile
import unittest
from pathlib import Path

from faceledger.comparison import (
    ComparisonRequest,
    ProgressNotification,
    compare,
)
from faceledger.presentation import present_comparison, render_matches


class RecordingRecognition:
    def __init__(self, vectors: dict[Path, tuple[float, ...]]) -> None:
        self._vectors = vectors
        self.calls: list[Path] = []

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        self.calls.append(image_path)
        return self._vectors[image_path]


class ComparisonCancellationTests(unittest.TestCase):
    def test_uncached_work_emits_typed_progress_separate_from_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "people"
            target_root.mkdir()
            target = target_root / "Alice.face1.jpg"
            target.write_bytes(b"target")
            observed: list[ProgressNotification] = []

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                RecordingRecognition(
                    {source: (1.0, 0.0), target: (1.0, 0.0)}
                ),
                on_progress=observed.append,
            )

        self.assertTrue(outcome.successful)
        self.assertTrue(outcome.complete)
        self.assertEqual(outcome.progress, tuple(observed))
        self.assertEqual(
            [(item.category, item.path) for item in observed],
            [("source", source), ("target", target)],
        )
        self.assertEqual([item.completed_items for item in observed], [1, 2])
        self.assertEqual(outcome.diagnostics, ())
        self.assertEqual(len(outcome.matches), 1)

    def test_cancellation_stops_at_the_next_item_boundary_and_hides_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "people"
            target_root.mkdir()
            first_target = target_root / "Alice.face1.jpg"
            first_target.write_bytes(b"first")
            unprocessed_target = target_root / "Bob.face2.jpg"
            unprocessed_target.write_bytes(b"second")
            recognition = RecordingRecognition(
                {
                    source: (1.0, 0.0),
                    first_target: (1.0, 0.0),
                    unprocessed_target: (1.0, 0.0),
                }
            )
            cancellation = {"requested": False}

            def observe(notification: ProgressNotification) -> None:
                if notification.category == "target":
                    cancellation["requested"] = True

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                recognition,
                on_progress=observe,
                cancellation_requested=lambda: cancellation["requested"],
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            status = present_comparison(outcome, stdout, stderr)

        self.assertFalse(outcome.successful)
        self.assertFalse(outcome.complete)
        self.assertEqual(outcome.matches, ())
        self.assertEqual(outcome.target_identities_compared, 1)
        self.assertEqual(outcome.diagnostics[-1].severity, "info")
        self.assertEqual(outcome.diagnostics[-1].category, "operation")
        self.assertEqual(outcome.diagnostics[-1].code, "comparison-cancelled")
        self.assertEqual(
            recognition.calls,
            [source, first_target],
        )
        self.assertEqual(render_matches(outcome), "")
        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("comparison-cancelled", stderr.getvalue())

    def test_cancellation_before_source_work_processes_no_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "people"
            target_root.mkdir()
            recognition = RecordingRecognition({source: (1.0, 0.0)})

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                recognition,
                cancellation_requested=lambda: True,
            )

        self.assertFalse(outcome.successful)
        self.assertFalse(outcome.complete)
        self.assertEqual(outcome.progress, ())
        self.assertEqual(recognition.calls, [])


if __name__ == "__main__":
    unittest.main()
