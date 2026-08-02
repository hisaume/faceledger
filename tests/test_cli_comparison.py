import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from faceledger.cli import main
from faceledger.comparison import RecognitionFailure
from faceledger.vector_profiles import VectorProfile


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


class ExplodingRecognition:
    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        raise RuntimeError("recognition backend exploded")


class RecognitionWithFailures(DeterministicRecognition):
    def __init__(
        self,
        vectors: dict[Path, tuple[float, ...]],
        failures: set[Path],
    ) -> None:
        super().__init__(vectors)
        self._failures = failures

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        if image_path in self._failures:
            self.calls.append((image_path, profile))
            raise RecognitionFailure("selected image has no usable face")
        return super().vector_for(image_path, profile)


class LiveDiagnosticRecognition(RecognitionWithFailures):
    def __init__(
        self,
        vectors: dict[Path, tuple[float, ...]],
        failure: Path,
        observe_at: Path,
        stderr: io.StringIO,
    ) -> None:
        super().__init__(vectors, {failure})
        self._observe_at = observe_at
        self._stderr = stderr
        self.warning_was_live = False

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        if image_path == self._observe_at:
            self.warning_was_live = "target-face-unusable" in self._stderr.getvalue()
        return super().vector_for(image_path, profile)


class StandaloneComparisonCliTests(unittest.TestCase):
    def test_compares_a_supported_jpeg_with_default_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source-without-an-image-extension"
            Image.new("RGB", (2, 2), "white").save(source, format="JPEG")
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "other.face0.jpg"
            target.write_bytes(b"target")
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        target: (-1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(status, 0)
            self.assertEqual(
                stdout.getvalue(),
                f"Source: {source}\n"
                f"Target root: {target_root}\n"
                "Model: Facenet512\n"
                "Threshold: 0.300000\n\n"
                "No matches found\n",
            )
            self.assertEqual(stderr.getvalue(), "")

    def test_maps_comparison_controls_to_the_core_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            root_target = target_root / "root.face0.png"
            root_target.write_bytes(b"root target")
            np.save(
                root_target.with_name(f"{root_target.name}.arcface.npy"),
                np.asarray((1.0,) + (0.0,) * 511),
                allow_pickle=False,
            )
            descendant = target_root / "descendant"
            descendant.mkdir()
            nested_target = descendant / "nested.face0.png"
            nested_target.write_bytes(b"nested target")
            recognition = DeterministicRecognition(
                {
                    source: (1.0,) + (0.0,) * 511,
                    root_target: (-1.0,) + (0.0,) * 511,
                    nested_target: (1.0,) + (0.0,) * 511,
                }
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                [
                    "compare",
                    str(source),
                    str(target_root),
                    "--model",
                    "arcface",
                    "--threshold",
                    "0.5",
                    "--no-cache",
                    "--no-recursive",
                    "--no-progress",
                ],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 0)
            self.assertIn("Model: ArcFace\n", stdout.getvalue())
            self.assertIn("Threshold: 0.500000\n", stdout.getvalue())
            self.assertIn("No matches found\n", stdout.getvalue())
            self.assertEqual(
                [path for path, _profile in recognition.calls],
                [source, root_target],
            )
            self.assertTrue(
                all(
                    profile.model_name == "ArcFace"
                    for _path, profile in recognition.calls
                )
            )
            self.assertEqual(stderr.getvalue(), "")

    def test_rejects_invalid_model_and_threshold_options_before_recognition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            recognition = DeterministicRecognition({})
            invalid_options = (
                ("--model", "Facenet512"),
                ("--threshold", "not-a-number"),
                ("--threshold", "nan"),
                ("--threshold", "inf"),
                ("--threshold", "-0.01"),
                ("--threshold", "2.01"),
            )

            for option in invalid_options:
                with self.subTest(option=option):
                    stdout = io.StringIO()
                    stderr = io.StringIO()

                    status = main(
                        [
                            "compare",
                            str(source),
                            str(target_root),
                            *option,
                        ],
                        stdout=stdout,
                        stderr=stderr,
                        recognition=recognition,
                    )

                    self.assertEqual(status, 2)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertIn("error:", stderr.getvalue())
            self.assertEqual(recognition.calls, [])

    def test_rejects_an_unsupported_source_by_content_before_recognition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "misleading.jpg"
            Image.new("RGB", (2, 2), "white").save(source, format="GIF")
            target_root = root / "face tree"
            target_root.mkdir()
            recognition = DeterministicRecognition({})
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("source-format-unsupported", stderr.getvalue())
            self.assertIn(
                "JPEG, PNG, or one-frame static WebP",
                stderr.getvalue(),
            )
            self.assertEqual(recognition.calls, [])

    def test_rejects_an_animated_webp_before_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "animated-source"
            frames = [
                Image.new("RGB", (2, 2), "red"),
                Image.new("RGB", (2, 2), "blue"),
            ]
            frames[0].save(
                source,
                format="WEBP",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )
            target_root = root / "face tree"
            target_root.mkdir()
            recognition = DeterministicRecognition({})
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("source-webp-animated", stderr.getvalue())
            self.assertIn("one-frame static WebP", stderr.getvalue())
            self.assertEqual(recognition.calls, [])

    def test_reports_an_unexpected_failure_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=ExplodingRecognition(),
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("internal-error", stderr.getvalue())
            self.assertIn("recognition backend exploded", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_accepts_static_webp_by_content_without_rewriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source-with-a-misleading-name.jpg"
            Image.new("RGB", (2, 2), "white").save(source, format="WEBP")
            original_bytes = source.read_bytes()
            target_root = root / "face tree"
            target_root.mkdir()
            recognition = DeterministicRecognition({source: (1.0,) + (0.0,) * 511})
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 0)
            self.assertIn("No matches found\n", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_accepts_inclusive_threshold_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            recognition = DeterministicRecognition({source: (1.0,) + (0.0,) * 511})

            for threshold, rendered in (("0", "0.000000"), ("2", "2.000000")):
                with self.subTest(threshold=threshold):
                    stdout = io.StringIO()
                    stderr = io.StringIO()

                    status = main(
                        [
                            "compare",
                            str(source),
                            str(target_root),
                            "--threshold",
                            threshold,
                        ],
                        stdout=stdout,
                        stderr=stderr,
                        recognition=recognition,
                    )

                    self.assertEqual(status, 0)
                    self.assertIn(
                        f"Threshold: {rendered}\n",
                        stdout.getvalue(),
                    )
                    self.assertEqual(stderr.getvalue(), "")

    def test_rejects_invalid_direct_source_paths_before_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target_root = root / "face tree"
            target_root.mkdir()
            corrupt_source = root / "corrupt.png"
            corrupt_source.write_bytes(b"not image data")
            missing_source = root / "missing.jpg"
            unreadable_source = root / "unreadable.jpg"
            Image.new("RGB", (2, 2), "white").save(unreadable_source)
            unreadable_source.chmod(0)
            recognition = DeterministicRecognition({})

            try:
                for source in (
                    corrupt_source,
                    missing_source,
                    unreadable_source,
                ):
                    with self.subTest(source=source):
                        stdout = io.StringIO()
                        stderr = io.StringIO()

                        status = main(
                            ["compare", str(source), str(target_root)],
                            stdout=stdout,
                            stderr=stderr,
                            recognition=recognition,
                        )

                        self.assertEqual(status, 1)
                        self.assertEqual(stdout.getvalue(), "")
                        self.assertIn("ERROR [input:", stderr.getvalue())
            finally:
                unreadable_source.chmod(0o600)
            self.assertEqual(recognition.calls, [])

    def test_keeps_warning_bearing_comparison_successful(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "unusable.face0.jpg"
            target.write_bytes(b"target")
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=RecognitionWithFailures(
                    {source: (1.0,) + (0.0,) * 511},
                    {target},
                ),
            )

            self.assertEqual(status, 0)
            self.assertIn("No matches found\n", stdout.getvalue())
            self.assertIn("WARNING [target:target-face-unusable]", stderr.getvalue())
            self.assertIn("WARNING SUMMARY: 1 warning", stderr.getvalue())

    def test_reports_a_missing_target_root_as_an_operation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            missing_target = root / "missing face tree"
            recognition = DeterministicRecognition({})
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source), str(missing_target)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("target-root-invalid", stderr.getvalue())
            self.assertEqual(recognition.calls, [])

    def test_presents_equal_distance_candidates_in_canonical_path_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            root_target = target_root / "Z.face0.jpg"
            root_target.write_bytes(b"root target")
            descendant = target_root / "A"
            descendant.mkdir()
            nested_target = descendant / "person.face0.jpg"
            nested_target.write_bytes(b"nested target")
            vector = (1.0,) + (0.0,) * 511
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition(
                    {
                        source: vector,
                        root_target: vector,
                        nested_target: vector,
                    }
                ),
            )

            self.assertEqual(status, 0)
            rendered = stdout.getvalue()
            self.assertLess(
                rendered.index("A/person.face0.jpg"),
                rendered.index("Z.face0.jpg"),
            )
            self.assertEqual(stderr.getvalue(), "")


class SourceFolderComparisonCliTests(unittest.TestCase):
    def test_requires_the_exact_lowercase_source_folder_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "source identity"
            source_folder.mkdir()
            (source_folder / "Folder.jpg").write_bytes(b"wrong-case anchor")
            target_root = root / "face tree"
            target_root.mkdir()
            recognition = DeterministicRecognition({})
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source_folder), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("source-folder-invalid", stderr.getvalue())
            self.assertIn("exact lowercase folder.jpg", stderr.getvalue())
            self.assertEqual(recognition.calls, [])

    def test_infers_an_anchored_source_folder_without_scanning_descendants(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "source identity"
            source_folder.mkdir()
            anchor = source_folder / "folder.jpg"
            anchor.write_bytes(b"anchor")
            second_source = source_folder / "folder1.jpg"
            second_source.write_bytes(b"second source")
            descendant = source_folder / "descendant"
            descendant.mkdir()
            (descendant / "folder2.jpg").write_bytes(b"not a source image")
            target_root = root / "face tree"
            target_root.mkdir()
            recognition = DeterministicRecognition(
                {
                    anchor: (1.0, 0.0),
                    second_source: (1.0, 0.0),
                }
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source_folder), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 0)
            self.assertIn(f"Source: {source_folder}\n", stdout.getvalue())
            self.assertIn("No matches found\n", stdout.getvalue())
            self.assertEqual(
                [path for path, _profile in recognition.calls],
                [anchor, second_source],
            )
            self.assertEqual(stderr.getvalue(), "")

    def test_streams_each_warning_once_before_later_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            unusable_target = target_root / "A.face0.jpg"
            unusable_target.write_bytes(b"unusable target")
            usable_target = target_root / "B.face1.jpg"
            usable_target.write_bytes(b"usable target")
            stderr = io.StringIO()
            recognition = LiveDiagnosticRecognition(
                {
                    source: (1.0, 0.0),
                    usable_target: (1.0, 0.0),
                },
                unusable_target,
                usable_target,
                stderr,
            )
            stdout = io.StringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 0)
            self.assertTrue(recognition.warning_was_live)
            self.assertEqual(stderr.getvalue().count("target-face-unusable"), 1)
            self.assertIn("WARNING SUMMARY: 1 warning", stderr.getvalue())
            self.assertIn("B.face1.jpg", stdout.getvalue())

    def test_clears_interactive_progress_around_a_live_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            unusable_target = target_root / "A.face0.jpg"
            unusable_target.write_bytes(b"unusable target")
            usable_target = target_root / "B.face1.jpg"
            usable_target.write_bytes(b"usable target")
            vector = (1.0, 0.0)
            stdout = io.StringIO()
            stderr = TtyStringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=RecognitionWithFailures(
                    {
                        source: vector,
                        usable_target: vector,
                    },
                    {unusable_target},
                ),
            )

            self.assertEqual(status, 0)
            transcript = stderr.getvalue()
            warning_start = transcript.index("WARNING [target:target-face-unusable]")
            self.assertEqual(transcript[warning_start - 1], "\r")
            self.assertEqual(transcript.count("target-face-unusable"), 1)
            self.assertTrue(transcript.endswith("1 candidate match.\n"))

    def test_reports_a_console_callback_failure_as_presentation_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            unusable_target = target_root / "person.face0.jpg"
            unusable_target.write_bytes(b"unusable target")
            stdout = io.StringIO()
            stderr = FailOnceStringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=RecognitionWithFailures(
                    {source: (1.0, 0.0)},
                    {unusable_target},
                ),
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("presentation-failure", stderr.getvalue())
            self.assertIn("simulated console failure", stderr.getvalue())
            self.assertNotIn("internal-error", stderr.getvalue())

    def test_keeps_useful_results_after_a_source_folder_image_warning(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "source identity"
            source_folder.mkdir()
            unusable_anchor = source_folder / "folder.jpg"
            unusable_anchor.write_bytes(b"unusable anchor")
            usable_source = source_folder / "folder1.jpg"
            usable_source.write_bytes(b"usable source")
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            vector = (1.0, 0.0)
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source_folder), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=RecognitionWithFailures(
                    {
                        usable_source: vector,
                        target: vector,
                    },
                    {unusable_anchor},
                ),
            )

            self.assertEqual(status, 0)
            self.assertIn("person.face0.jpg", stdout.getvalue())
            self.assertEqual(
                stderr.getvalue().count("source-folder-image-unusable"),
                1,
            )
            self.assertIn("WARNING SUMMARY: 1 warning", stderr.getvalue())

    def test_fatal_source_folder_failure_stops_before_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "source identity"
            source_folder.mkdir()
            anchor = source_folder / "folder.jpg"
            anchor.write_bytes(b"unusable anchor")
            second_source = source_folder / "folder1.jpg"
            second_source.write_bytes(b"unusable source")
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            recognition = RecognitionWithFailures(
                {},
                {anchor, second_source},
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            status = main(
                ["compare", str(source_folder), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=recognition,
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue().count("source-folder-image-unusable"),
                2,
            )
            self.assertEqual(stderr.getvalue().count("source-folder-unusable"), 1)
            self.assertEqual(
                [path for path, _profile in recognition.calls],
                [anchor, second_source],
            )

    def test_shows_and_clears_transient_progress_on_interactive_stderr(
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
            vector = (1.0, 0.0)
            stdout = io.StringIO()
            stderr = TtyStringIO()

            status = main(
                ["compare", str(source), str(target_root)],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition(
                    {
                        source: vector,
                        target: vector,
                    }
                ),
            )

            self.assertEqual(status, 0)
            progress = stderr.getvalue()
            self.assertIn(f"Completed 1: {source}", progress)
            self.assertIn(f"Completed 2: {target}", progress)
            self.assertIn("\r", progress)
            self.assertTrue(progress.endswith("\r"))
            self.assertNotIn("%", progress)
            self.assertNotIn("ETA", progress)
            self.assertNotIn("Completed", stdout.getvalue())

    def test_no_progress_suppresses_interactive_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            target = target_root / "person.face0.jpg"
            target.write_bytes(b"target")
            vector = (1.0, 0.0)
            stdout = io.StringIO()
            stderr = TtyStringIO()

            status = main(
                ["compare", str(source), str(target_root), "--no-progress"],
                stdout=stdout,
                stderr=stderr,
                recognition=DeterministicRecognition(
                    {
                        source: vector,
                        target: vector,
                    }
                ),
            )

            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("person.face0.jpg", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
