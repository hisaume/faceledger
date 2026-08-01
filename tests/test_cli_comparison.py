import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from faceledger.cli import main
from faceledger.comparison import RecognitionFailure
from faceledger.vector_profiles import VectorProfile


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
            source_directory = root / "source directory"
            source_directory.mkdir()
            corrupt_source = root / "corrupt.png"
            corrupt_source.write_bytes(b"not image data")
            missing_source = root / "missing.jpg"
            unreadable_source = root / "unreadable.jpg"
            Image.new("RGB", (2, 2), "white").save(unreadable_source)
            unreadable_source.chmod(0)
            recognition = DeterministicRecognition({})

            try:
                for source in (
                    source_directory,
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


if __name__ == "__main__":
    unittest.main()
