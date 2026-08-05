import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from faceledger.comparison import (
    CandidateMatch,
    ComparisonMetadata,
    ComparisonOutcome,
    Diagnostic,
)
from faceledger.presentation import (
    ComparisonArtifactRequest,
    write_comparison_artifacts,
)


class ComparisonArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.outcome = ComparisonOutcome(
            matches=(CandidateMatch(Path("People/Alice"), 0.125),),
            diagnostics=(),
            target_identities_compared=3,
            metadata=ComparisonMetadata(
                source=Path("/photos/source.jpg"),
                target_root=Path("/photos/people"),
                model_name="Facenet512",
                threshold=0.3,
            ),
        )

    def test_no_destinations_leave_the_filesystem_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            returned = write_comparison_artifacts(
                self.outcome,
                ComparisonArtifactRequest(),
            )

            self.assertIs(returned, self.outcome)
            self.assertEqual(tuple(root.iterdir()), ())

    def test_requested_result_contains_the_resolved_human_readable_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "comparison.txt"

            returned = write_comparison_artifacts(
                self.outcome,
                ComparisonArtifactRequest(result_path=result_path),
            )

            self.assertTrue(returned.successful)
            self.assertEqual(
                result_path.read_text(encoding="utf-8"),
                "Source: /photos/source.jpg\n"
                "Target root: /photos/people\n"
                "Model: Facenet512\n"
                "Threshold: 0.300000\n\n"
                "Rank  Identity      Cosine distance\n"
                "1     People/Alice  0.125000\n",
            )

    def test_incomplete_comparison_writes_no_result_artifact(self) -> None:
        incomplete = replace(
            self.outcome,
            matches=(),
            successful=True,
            complete=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "comparison.txt"

            returned = write_comparison_artifacts(
                incomplete,
                ComparisonArtifactRequest(result_path=result_path),
            )

            self.assertIs(returned, incomplete)
            self.assertFalse(result_path.exists())

    def test_cancelled_comparison_writes_a_log_but_no_result(self) -> None:
        cancelled = replace(
            self.outcome,
            matches=(),
            diagnostics=(
                Diagnostic(
                    severity="info",
                    category="operation",
                    code="comparison-cancelled",
                    path=None,
                    message="Comparison cancelled; the operation is incomplete.",
                ),
            ),
            successful=False,
            complete=False,
            target_identities_compared=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_path = root / "comparison.txt"
            log_path = root / "comparison.log"

            returned = write_comparison_artifacts(
                cancelled,
                ComparisonArtifactRequest(
                    result_path=result_path,
                    log_path=log_path,
                ),
            )

            self.assertIs(returned, cancelled)
            self.assertFalse(result_path.exists())
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("Status: cancelled\n", log)
            self.assertIn("Target identities compared: 2\n", log)
            self.assertEqual(log.count("comparison-cancelled"), 1)

    def test_requested_log_contains_metadata_diagnostics_and_counts_not_results(
        self,
    ) -> None:
        outcome = ComparisonOutcome(
            matches=self.outcome.matches,
            diagnostics=(
                Diagnostic(
                    severity="warning",
                    category="target",
                    code="target-face-unusable",
                    path=Path("/photos/people/Broken.face1.jpg"),
                    message="Expected one face but found two.",
                ),
            ),
            target_identities_compared=3,
            metadata=self.outcome.metadata,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "comparison.log"

            returned = write_comparison_artifacts(
                outcome,
                ComparisonArtifactRequest(log_path=log_path),
            )

            self.assertTrue(returned.successful)
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("Source: /photos/source.jpg", log_text)
            self.assertIn("Target root: /photos/people", log_text)
            self.assertIn("Model: Facenet512", log_text)
            self.assertIn("Threshold: 0.300000", log_text)
            self.assertIn("Status: successful", log_text)
            self.assertIn("Target identities compared: 3", log_text)
            self.assertIn("Candidate matches: 1", log_text)
            self.assertIn("Diagnostics: 1", log_text)
            self.assertIn(
                "WARNING [target:target-face-unusable] "
                "/photos/people/Broken.face1.jpg: "
                "Expected one face but found two.",
                log_text,
            )
            self.assertNotIn("Rank", log_text)
            self.assertNotIn("People/Alice", log_text)

    def test_result_write_failure_is_an_operation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "directory"
            destination.mkdir()

            returned = write_comparison_artifacts(
                self.outcome,
                ComparisonArtifactRequest(result_path=destination),
            )

            self.assertFalse(returned.successful)
            self.assertEqual(returned.diagnostics[-1].severity, "error")
            self.assertEqual(returned.diagnostics[-1].category, "output")
            self.assertEqual(
                returned.diagnostics[-1].code,
                "result-artifact-write-failed",
            )
            self.assertEqual(returned.diagnostics[-1].path, destination)

    def test_log_write_failure_is_an_operation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "directory"
            destination.mkdir()

            returned = write_comparison_artifacts(
                self.outcome,
                ComparisonArtifactRequest(log_path=destination),
            )

            self.assertFalse(returned.successful)
            self.assertEqual(returned.diagnostics[-1].severity, "error")
            self.assertEqual(returned.diagnostics[-1].category, "output")
            self.assertEqual(
                returned.diagnostics[-1].code,
                "log-artifact-write-failed",
            )
            self.assertEqual(returned.diagnostics[-1].path, destination)


if __name__ == "__main__":
    unittest.main()
