import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from faceledger.cli import main
from faceledger.comparison import RecognitionFailure
from faceledger.vector_profiles import VectorProfile


def unit_vector(index: int = 0) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(512))


class DeterministicRecognition:
    def __init__(self, vectors: dict[Path, tuple[float, ...]]) -> None:
        self._vectors = vectors
        self.calls: list[tuple[Path, VectorProfile]] = []

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        self.calls.append((image_path, profile))
        return self._vectors[image_path]


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class FailOnceStringIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    def write(self, text: str) -> int:
        if not self._failed:
            self._failed = True
            raise OSError("simulated console failure")
        return super().write(text)


class ExplodingRecognition:
    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        raise RuntimeError("recognition backend exploded")


class LiveDiagnosticRecognition(DeterministicRecognition):
    def __init__(
        self,
        vectors: dict[Path, tuple[float, ...]],
        *,
        failure: Path,
        observe_at: Path,
        stderr: io.StringIO,
    ) -> None:
        super().__init__(vectors)
        self._failure = failure
        self._observe_at = observe_at
        self._stderr = stderr
        self.warning_was_live = False

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        if image_path == self._observe_at:
            self.warning_was_live = (
                "cache-build-image-unusable" in self._stderr.getvalue()
            )
        if image_path == self._failure:
            raise RecognitionFailure("selected image has no usable face")
        return super().vector_for(image_path, profile)


class CacheMaintenanceGrammarTests(unittest.TestCase):
    def test_nested_help_describes_the_maintenance_contract(self) -> None:
        for operation in ("build", "rebuild"):
            with self.subTest(operation=operation):
                stdout = io.StringIO()
                stderr = io.StringIO()

                status = main(
                    ["cache", operation, "--help"],
                    stdout=stdout,
                    stderr=stderr,
                )

                self.assertEqual(status, 0)
                help_text = stdout.getvalue()
                self.assertIn("maintenance_root", help_text)
                self.assertIn("--model {facenet512,arcface}", help_text)
                self.assertIn("--recursive", help_text)
                self.assertIn("--no-progress", help_text)
                self.assertEqual(stderr.getvalue(), "")

    def test_cache_requires_a_nested_operation(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        status = main(["cache"], stdout=stdout, stderr=stderr)

        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(
            "the following arguments are required: operation", stderr.getvalue()
        )

    def test_model_choices_accept_only_lowercase_cli_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["cache", "build", str(root), "--model", "Facenet512"],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(status, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("invalid choice: 'Facenet512'", stderr.getvalue())


class CacheBuildCliTests(unittest.TestCase):
    def test_builds_default_model_cache_and_reports_completed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            descendant = root / "Descendant"
            descendant.mkdir()
            descendant_face = descendant / "Nested.face1.jpg"
            descendant_face.write_bytes(b"nested face")
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["cache", "build", str(root)],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition({face: unit_vector()}),
            )

            self.assertEqual(status, 0)
            self.assertTrue((root / "Person.face0.jpg.facenet512.npy").is_file())
            self.assertFalse((descendant / "Nested.face1.jpg.facenet512.npy").exists())
            self.assertEqual(
                stdout.getvalue(),
                "Operation: cache build\n"
                "Status: successful\n"
                f"Root: {root}\n"
                "Model: Facenet512\n"
                "Scope: selected root\n"
                "Created: 1\n"
                "Retained: 0\n",
            )
            self.assertEqual(stderr.getvalue(), "")

    def test_recursive_build_expands_scope_and_reports_retained_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            root_face = root / "Person.face0.jpg"
            root_face.write_bytes(b"root face")
            root_cache = root / "Person.face0.jpg.facenet512.npy"
            np.save(root_cache, np.asarray(unit_vector()))
            descendant = root / "Descendant"
            descendant.mkdir()
            descendant_face = descendant / "Nested.face1.jpg"
            descendant_face.write_bytes(b"nested face")
            descendant_cache = descendant / "Nested.face1.jpg.facenet512.npy"
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["cache", "build", str(root), "--recursive"],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition({descendant_face: unit_vector(1)}),
            )

            self.assertEqual(status, 0)
            self.assertTrue(descendant_cache.is_file())
            self.assertIn("Scope: recursive\n", stdout.getvalue())
            self.assertIn("Created: 1\n", stdout.getvalue())
            self.assertIn("Retained: 1\n", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_warning_is_live_once_and_does_not_change_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            unusable = root / "A.face0.jpg"
            unusable.write_bytes(b"unusable face")
            usable = root / "B.face1.jpg"
            usable.write_bytes(b"usable face")
            stdout = io.StringIO()
            stderr = io.StringIO()
            recognition = LiveDiagnosticRecognition(
                {usable: unit_vector()},
                failure=unusable,
                observe_at=usable,
                stderr=stderr,
            )

            status = main(
                ["cache", "build", str(root)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 0)
            self.assertTrue(recognition.warning_was_live)
            self.assertIn("Status: successful\n", stdout.getvalue())
            self.assertIn("Created: 1\n", stdout.getvalue())
            transcript = stderr.getvalue()
            self.assertEqual(transcript.count("cache-build-image-unusable"), 1)
            self.assertEqual(transcript.count("WARNING SUMMARY: 1 warning."), 1)

    def test_complete_operation_failure_still_reports_incomplete_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Missing Face Tree"
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["cache", "build", str(root)],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition({}),
            )

            self.assertEqual(status, 1)
            self.assertEqual(
                stdout.getvalue(),
                "Operation: cache build\n"
                "Status: incomplete\n"
                f"Root: {root}\n"
                "Model: Facenet512\n"
                "Scope: selected root\n"
                "Created: 0\n"
                "Retained: 0\n",
            )
            self.assertEqual(stderr.getvalue().count("maintenance-root-invalid"), 1)

    def test_interactive_progress_is_cleared_after_the_last_item(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            stdout = io.StringIO()
            stderr = TtyStringIO()

            status = main(
                ["cache", "build", str(root)],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition({face: unit_vector()}),
            )

            self.assertEqual(status, 0)
            transcript = stderr.getvalue()
            progress = f"Completed 1: {face}"
            self.assertIn(f"\r{progress}", transcript)
            self.assertTrue(transcript.endswith(f"\r{' ' * len(progress)}\r"))

    def test_no_progress_suppresses_interactive_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            stdout = io.StringIO()
            stderr = TtyStringIO()

            status = main(
                ["cache", "build", str(root), "--no-progress"],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition({face: unit_vector()}),
            )

            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")

    def test_console_callback_failure_is_not_reported_as_internal_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"unusable face")
            stdout = io.StringIO()
            stderr = FailOnceStringIO()

            status = main(
                ["cache", "build", str(root)],
                stdout=stdout,
                stderr=stderr,
                recognition=LiveDiagnosticRecognition(
                    {},
                    failure=face,
                    observe_at=root / "not-reached.jpg",
                    stderr=stderr,
                ),
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("presentation-failure", stderr.getvalue())
            self.assertIn("simulated console failure", stderr.getvalue())
            self.assertNotIn("internal-error", stderr.getvalue())

    def test_unexpected_failure_is_concise_and_reports_incomplete_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            (root / "Person.face0.jpg").write_bytes(b"face")
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["cache", "build", str(root)],
                stdout=stdout,
                stderr=stderr,
                recognition=ExplodingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertIn("Status: incomplete\n", stdout.getvalue())
            self.assertIn("Created: 0\n", stdout.getvalue())
            self.assertEqual(stderr.getvalue().count("internal-error"), 1)
            self.assertIn("recognition backend exploded", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


class CacheRebuildCliTests(unittest.TestCase):
    def test_rebuilds_only_the_lowercase_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            facenet_cache = root / "Person.face0.jpg.facenet512.npy"
            arcface_cache = root / "Person.face0.jpg.arcface.npy"
            np.save(facenet_cache, np.asarray(unit_vector(0)))
            np.save(arcface_cache, np.asarray(unit_vector(1)))
            facenet_before = facenet_cache.read_bytes()
            recognition = DeterministicRecognition({face: unit_vector(2)})
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["cache", "rebuild", str(root), "--model", "arcface"],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 0)
            self.assertEqual(facenet_cache.read_bytes(), facenet_before)
            np.testing.assert_array_equal(np.load(arcface_cache), unit_vector(2))
            self.assertEqual(recognition.calls[0][1].model_name, "ArcFace")
            self.assertEqual(
                stdout.getvalue(),
                "Operation: cache rebuild\n"
                "Status: successful\n"
                f"Root: {root}\n"
                "Model: ArcFace\n"
                "Scope: selected root\n"
                "Rebuilt: 1\n",
            )
            self.assertEqual(stderr.getvalue(), "")

    def test_failed_recalculation_preserves_the_previous_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"unusable face")
            cache = root / "Person.face0.jpg.facenet512.npy"
            np.save(cache, np.asarray(unit_vector()))
            before = cache.read_bytes()
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["cache", "rebuild", str(root)],
                stdout=stdout,
                stderr=stderr,
                recognition=LiveDiagnosticRecognition(
                    {},
                    failure=face,
                    observe_at=root / "not-reached.jpg",
                    stderr=stderr,
                ),
            )

            self.assertEqual(status, 0)
            self.assertEqual(cache.read_bytes(), before)
            self.assertIn("Status: successful\n", stdout.getvalue())
            self.assertIn("Rebuilt: 0\n", stdout.getvalue())
            self.assertEqual(stderr.getvalue().count("cache-rebuild-image-unusable"), 1)
            self.assertIn("WARNING SUMMARY: 1 warning.", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
