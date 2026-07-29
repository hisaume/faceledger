import io
import tempfile
import unittest
from pathlib import Path

from faceledger.comparison import (
    ComparisonRequest,
    RecognitionFailure,
    compare,
)
from faceledger.presentation import present_comparison


class RecognitionWithFailures:
    def __init__(
        self,
        vectors: dict[Path, tuple[float, ...]],
        failures: dict[Path, str] | None = None,
    ) -> None:
        self._vectors = vectors
        self._failures = failures or {}

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        if image_path in self._failures:
            raise RecognitionFailure(self._failures[image_path])
        return self._vectors[image_path]


class ComparisonPresentationTests(unittest.TestCase):
    def test_keeps_results_on_stdout_and_reports_every_warning_on_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Face Tree"
            target_root.mkdir()
            first_bad_face = target_root / "Group.face1.jpg"
            first_bad_face.write_bytes(b"multiple faces")
            second_bad_face = target_root / "Broken.face2.png"
            second_bad_face.write_bytes(b"broken image")
            usable_face = target_root / "Candidate.face3.webp"
            usable_face.write_bytes(b"one face")
            request = ComparisonRequest(source=source, target_root=target_root)
            outcome = compare(
                request,
                RecognitionWithFailures(
                    vectors={
                        source: (1.0, 0.0),
                        usable_face: (1.0, 0.0),
                    },
                    failures={
                        first_bad_face: "Expected one face but found two",
                        second_bad_face: "Image could not be loaded",
                    },
                ),
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = present_comparison(outcome, stdout, stderr)

        result_text = stdout.getvalue()
        diagnostic_text = stderr.getvalue()
        self.assertEqual(status, 0)
        self.assertTrue(outcome.successful)
        self.assertEqual(len(outcome.matches), 1)
        self.assertEqual(len(outcome.diagnostics), 2)
        for diagnostic in outcome.diagnostics:
            self.assertEqual(diagnostic.severity, "warning")
            self.assertEqual(diagnostic.category, "target")
            self.assertEqual(diagnostic.code, "target-face-unusable")
            self.assertIsNotNone(diagnostic.path)
            self.assertTrue(diagnostic.message)

        self.assertIn(f"Source: {source.resolve()}", result_text)
        self.assertIn(f"Target root: {target_root.resolve()}", result_text)
        self.assertIn("Model: Facenet512", result_text)
        self.assertIn("Threshold: 0.300000", result_text)
        self.assertIn("Rank", result_text)
        self.assertIn("Identity", result_text)
        self.assertIn("Cosine distance", result_text)
        self.assertIn("Candidate.face3.webp  0.000000", result_text)
        self.assertNotIn("WARNING", result_text)
        self.assertNotIn("target-face-unusable", result_text)

        self.assertEqual(
            diagnostic_text.count("WARNING [target:target-face-unusable]"), 2
        )
        self.assertIn(str(first_bad_face), diagnostic_text)
        self.assertIn("Expected one face but found two", diagnostic_text)
        self.assertIn(str(second_bad_face), diagnostic_text)
        self.assertIn("Image could not be loaded", diagnostic_text)
        self.assertIn("WARNING SUMMARY", diagnostic_text)
        self.assertIn("2 warnings", diagnostic_text)
        self.assertIn("1 target identity compared", diagnostic_text)
        self.assertIn("1 candidate match", diagnostic_text)
        self.assertNotIn("Rank  Identity", diagnostic_text)

    def test_successful_empty_result_has_zero_status_and_no_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Target Person"
            target_root.mkdir()
            target = target_root / "folder.jpg"
            target.write_bytes(b"target")
            request = ComparisonRequest(source=source, target_root=target_root)
            outcome = compare(
                request,
                RecognitionWithFailures(
                    vectors={
                        source: (1.0, 0.0),
                        target: (0.0, 1.0),
                    }
                ),
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = present_comparison(outcome, stdout, stderr)

        self.assertEqual(status, 0)
        self.assertTrue(outcome.successful)
        self.assertIn("No matches found", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_header_keeps_the_target_path_resolved_by_the_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            original_target = root / "Original Face Tree"
            original_target.mkdir()
            target = original_target / "Candidate.face0.jpg"
            target.write_bytes(b"target")
            selected_target = root / "Selected Face Tree"
            selected_target.symlink_to(original_target, target_is_directory=True)
            request = ComparisonRequest(source=source, target_root=selected_target)
            outcome = compare(
                request,
                RecognitionWithFailures(
                    vectors={
                        source: (1.0, 0.0),
                        target: (1.0, 0.0),
                    }
                ),
            )

            replacement_target = root / "Replacement Face Tree"
            replacement_target.mkdir()
            selected_target.unlink()
            selected_target.symlink_to(replacement_target, target_is_directory=True)
            stdout = io.StringIO()

            present_comparison(outcome, stdout, io.StringIO())

        self.assertIn(
            f"Target root: {original_target.resolve()}",
            stdout.getvalue(),
        )
        self.assertNotIn(str(replacement_target.resolve()), stdout.getvalue())

    def test_operation_error_has_nonzero_status_and_no_result_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "missing-source.jpg"
            target_root = root / "Face Tree"
            target_root.mkdir()
            request = ComparisonRequest(source=source, target_root=target_root)
            outcome = compare(request, RecognitionWithFailures(vectors={}))
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = present_comparison(outcome, stdout, stderr)

        self.assertEqual(status, 1)
        self.assertFalse(outcome.successful)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("ERROR [source:source-image-invalid]", stderr.getvalue())
        self.assertIn(str(source.resolve()), stderr.getvalue())
        self.assertIn("not a readable file", stderr.getvalue())
        self.assertNotIn("WARNING SUMMARY", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
