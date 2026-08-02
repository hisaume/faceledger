import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from faceledger.cli import main
from faceledger.comparison import RecognitionFailure
from faceledger.vector_profiles import VectorProfile


class RecordingRecognition:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        self.calls.append(image_path)
        return (1.0,) + (0.0,) * (profile.expected_dimensions - 1)


class ArtifactTimingRecognition(RecordingRecognition):
    def __init__(self, destination: Path) -> None:
        super().__init__()
        self._destination = destination
        self.observed_contents: list[str | None] = []

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        content = (
            self._destination.read_text(encoding="utf-8")
            if self._destination.exists()
            else None
        )
        self.observed_contents.append(content)
        return super().vector_for(image_path, profile)


class FailingRecognition:
    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        raise RecognitionFailure("selected source has no usable face")


class TargetOutcomeRecognition(RecordingRecognition):
    def __init__(self, target: Path, *, fail_target: bool = False) -> None:
        super().__init__()
        self._target = target
        self._fail_target = fail_target

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        if image_path == self._target:
            self.calls.append(image_path)
            if self._fail_target:
                raise RecognitionFailure("selected target has no usable face")
            return (-1.0,) + (0.0,) * (profile.expected_dimensions - 1)
        return super().vector_for(image_path, profile)


class ComparisonArtifactCliTests(unittest.TestCase):
    def test_compare_help_lists_both_artifact_destinations(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        status = main(["compare", "--help"], stdout=stdout, stderr=stderr)

        self.assertEqual(status, 0)
        self.assertIn("--result-file RESULT_FILE", stdout.getvalue())
        self.assertIn("--log-file LOG_FILE", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_rejects_normalized_alias_and_hard_link_collisions_before_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "missing-source.jpg"
            normalized_result = root / "normalized.txt"
            alias_target = root / "alias-target.txt"
            alias_target.write_text("existing", encoding="utf-8")
            alias = root / "alias.txt"
            alias.symlink_to(alias_target)
            hard_link_target = root / "hard-link-target.txt"
            hard_link_target.write_text("existing", encoding="utf-8")
            hard_link = root / "hard-link.txt"
            hard_link.hardlink_to(hard_link_target)
            collisions = (
                (
                    normalized_result,
                    root / "unused" / ".." / normalized_result.name,
                ),
                (alias_target, alias),
                (hard_link_target, hard_link),
            )

            for result_path, log_path in collisions:
                with self.subTest(result_path=result_path, log_path=log_path):
                    recognition = RecordingRecognition()
                    stdout = io.StringIO()
                    stderr = io.StringIO()

                    status = main(
                        [
                            "compare",
                            str(source),
                            str(root),
                            "--result-file",
                            str(result_path),
                            "--log-file",
                            str(log_path),
                        ],
                        stdout=stdout,
                        stderr=stderr,
                        recognition=recognition,
                    )

                    self.assertEqual(status, 2)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertIn("must identify different files", stderr.getvalue())
                    self.assertNotIn("source-file-invalid", stderr.getvalue())
                    self.assertEqual(recognition.calls, [])

    def test_reports_a_cyclic_artifact_symlink_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            result_path = root / "cyclic-result.txt"
            result_path.symlink_to(result_path.name)
            log_path = root / "comparison.log"
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                    "--log-file",
                    str(log_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=RecordingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertIn("No matches found\n", stdout.getvalue())
            self.assertEqual(
                stderr.getvalue().count("result-artifact-write-failed"),
                1,
            )
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIn(
                "Status: unsuccessful\n", log_path.read_text(encoding="utf-8")
            )

    def test_overwrites_a_requested_result_with_the_complete_stdout_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            result_path = root / "comparison.txt"
            result_path.write_text("stale result", encoding="utf-8")
            unrequested_log = root / "comparison.log"
            stdout = io.StringIO()
            stderr = io.StringIO()
            recognition = ArtifactTimingRecognition(result_path)

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 0)
            self.assertEqual(
                result_path.read_text(encoding="utf-8"),
                stdout.getvalue(),
            )
            self.assertIn("person.face0.jpg", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")
            self.assertFalse(unrequested_log.exists())
            self.assertEqual(
                recognition.observed_contents,
                ["stale result", "stale result"],
            )

    def test_writes_a_requested_result_when_no_candidates_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            result_path = root / "comparison.txt"
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=TargetOutcomeRecognition(target),
            )

            self.assertEqual(status, 0)
            self.assertIn("No matches found\n", stdout.getvalue())
            self.assertEqual(
                result_path.read_text(encoding="utf-8"),
                stdout.getvalue(),
            )
            self.assertEqual(stderr.getvalue(), "")

    def test_writes_a_requested_result_after_a_best_effort_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            result_path = root / "comparison.txt"
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=TargetOutcomeRecognition(target, fail_target=True),
            )

            self.assertEqual(status, 0)
            self.assertIn("No matches found\n", stdout.getvalue())
            self.assertEqual(
                result_path.read_text(encoding="utf-8"),
                stdout.getvalue(),
            )
            self.assertEqual(stderr.getvalue().count("target-face-unusable"), 1)
            self.assertIn("WARNING SUMMARY: 1 warning", stderr.getvalue())

    def test_writes_a_troubleshooting_log_without_results_or_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            log_path = root / "comparison.log"
            log_path.write_text("stale log", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--log-file",
                    str(log_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=RecordingRecognition(),
            )

            self.assertEqual(status, 0)
            log = log_path.read_text(encoding="utf-8")
            self.assertIn(f"Source: {source}\n", log)
            self.assertIn(f"Target root: {target_root}\n", log)
            self.assertIn("Model: Facenet512\n", log)
            self.assertIn("Threshold: 0.300000\n", log)
            self.assertIn("Status: successful\n", log)
            self.assertIn("Target identities compared: 1\n", log)
            self.assertIn("Candidate matches: 1\n", log)
            self.assertIn("Diagnostics: 0\n", log)
            self.assertNotIn("person.face0.jpg", log)
            self.assertNotIn("Rank", log)
            self.assertNotIn("Completed", log)
            self.assertIn("person.face0.jpg", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_result_write_failure_preserves_stdout_and_creates_no_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            missing_parent = root / "missing"
            result_path = missing_parent / "comparison.txt"
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=RecordingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertIn("person.face0.jpg", stdout.getvalue())
            self.assertEqual(
                stderr.getvalue().count("result-artifact-write-failed"),
                1,
            )
            self.assertIn(str(result_path), stderr.getvalue())
            self.assertFalse(missing_parent.exists())

    def test_rejects_an_existing_non_regular_artifact_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            result_path = root / "comparison.pipe"
            os.mkfifo(result_path)
            reader = os.open(result_path, os.O_RDONLY | os.O_NONBLOCK)
            stdout = io.StringIO()
            stderr = io.StringIO()

            try:
                status = main(
                    [
                        "compare",
                        str(source),
                        str(target_root),
                        "--result-file",
                        str(result_path),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    recognition=RecordingRecognition(),
                )
            finally:
                os.close(reader)

            self.assertEqual(status, 1)
            self.assertIn("person.face0.jpg", stdout.getvalue())
            self.assertEqual(
                stderr.getvalue().count("result-artifact-write-failed"),
                1,
            )
            self.assertTrue(result_path.exists())
            self.assertFalse(result_path.is_file())

    def test_rejects_a_dangling_artifact_symlink_without_creating_its_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            dangling_target = root / "unrequested-target.txt"
            result_path = root / "comparison.txt"
            result_path.symlink_to(dangling_target)
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=RecordingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertIn("No matches found\n", stdout.getvalue())
            self.assertEqual(
                stderr.getvalue().count("result-artifact-write-failed"),
                1,
            )
            self.assertTrue(result_path.is_symlink())
            self.assertFalse(dangling_target.exists())

    def test_log_write_failure_is_an_operation_error_without_hiding_stdout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            log_path = root / "log destination"
            log_path.mkdir()
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--log-file",
                    str(log_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=RecordingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertIn("person.face0.jpg", stdout.getvalue())
            self.assertEqual(
                stderr.getvalue().count("log-artifact-write-failed"),
                1,
            )
            self.assertIn("ERROR [output:log-artifact-write-failed]", stderr.getvalue())

    def test_logs_direct_source_validation_without_inventing_core_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "unsupported.jpg"
            Image.new("RGB", (2, 2), "white").save(source, format="GIF")
            target_root = root / "face tree"
            target_root.mkdir()
            log_path = root / "comparison.log"
            recognition = RecordingRecognition()
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--log-file",
                    str(log_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue().count("source-format-unsupported"), 1)
            self.assertEqual(recognition.calls, [])
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("Operation metadata: unavailable\n", log)
            self.assertIn("Status: unsuccessful\n", log)
            self.assertIn("Target identities compared: 0\n", log)
            self.assertIn("Candidate matches: 0\n", log)
            self.assertIn("Diagnostics: 1\n", log)
            self.assertEqual(log.count("source-format-unsupported"), 1)

    def test_failed_comparison_writes_only_the_requested_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            result_path = root / "comparison.txt"
            log_path = root / "comparison.log"
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                    "--log-file",
                    str(log_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=FailingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(result_path.exists())
            self.assertEqual(stderr.getvalue().count("source-image-unusable"), 1)
            log = log_path.read_text(encoding="utf-8")
            self.assertIn(f"Source: {source}\n", log)
            self.assertIn(f"Target root: {target_root}\n", log)
            self.assertIn("Status: unsuccessful\n", log)
            self.assertEqual(log.count("source-image-unusable"), 1)

    def test_log_records_a_result_artifact_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            result_path = root / "result destination"
            result_path.mkdir()
            log_path = root / "comparison.log"
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                    "--log-file",
                    str(log_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=RecordingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertIn("person.face0.jpg", stdout.getvalue())
            self.assertEqual(
                stderr.getvalue().count("result-artifact-write-failed"),
                1,
            )
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("Status: unsuccessful\n", log)
            self.assertIn("Candidate matches: 1\n", log)
            self.assertEqual(log.count("result-artifact-write-failed"), 1)

    def test_reports_simultaneous_result_and_log_failures_once_each(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            result_path = root / "result destination"
            result_path.mkdir()
            log_path = root / "log destination"
            log_path.mkdir()
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--result-file",
                    str(result_path),
                    "--log-file",
                    str(log_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=RecordingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertIn("person.face0.jpg", stdout.getvalue())
            self.assertEqual(
                stderr.getvalue().count("result-artifact-write-failed"),
                1,
            )
            self.assertEqual(
                stderr.getvalue().count("log-artifact-write-failed"),
                1,
            )

    def test_reports_validation_and_log_failures_once_each(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "missing-source.jpg"
            target_root = root / "face tree"
            target_root.mkdir()
            log_path = root / "log destination"
            log_path.mkdir()
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--log-file",
                    str(log_path),
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=RecordingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue().count("source-file-invalid"), 1)
            self.assertEqual(
                stderr.getvalue().count("log-artifact-write-failed"),
                1,
            )


if __name__ == "__main__":
    unittest.main()
