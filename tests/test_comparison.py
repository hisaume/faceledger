import tempfile
import unittest
from pathlib import Path

from faceledger.comparison import (
    CandidateMatch,
    ComparisonRequest,
    RecognitionFailure,
    compare,
)
from faceledger.presentation import render_matches


class DeterministicRecognition:
    def __init__(self, vectors: dict[Path, tuple[float, ...]]) -> None:
        self._vectors = vectors

    def vector_for(self, image_path: Path) -> tuple[float, ...]:
        return self._vectors[image_path]


class FailingRecognition:
    def vector_for(self, image_path: Path) -> tuple[float, ...]:
        raise RecognitionFailure(f"No usable face found in {image_path.name}")


class RecognitionWithFailures:
    def __init__(
        self,
        vectors: dict[Path, tuple[float, ...]],
        failures: dict[Path, str],
    ) -> None:
        self._vectors = vectors
        self._failures = failures

    def vector_for(self, image_path: Path) -> tuple[float, ...]:
        if image_path in self._failures:
            raise RecognitionFailure(self._failures[image_path])
        return self._vectors[image_path]


def snapshot_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class StandaloneComparisonTests(unittest.TestCase):
    def test_rejects_a_missing_source_before_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_root = Path(temporary_directory) / "Target Person"
            target_root.mkdir()

            outcome = compare(
                ComparisonRequest(target_root=target_root),
                DeterministicRecognition({}),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.matches, ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["source-selection-required"],
            )

    def test_rejects_ambiguous_source_selection_before_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_image = root / "source.jpg"
            source_image.write_bytes(b"source image")
            source_folder = root / "Source Person"
            source_folder.mkdir()
            (source_folder / "folder.jpg").write_bytes(b"folder source image")
            target_root = root / "Target Person"
            target_root.mkdir()

            outcome = compare(
                ComparisonRequest(
                    source=source_image,
                    source_folder=source_folder,
                    target_root=target_root,
                ),
                DeterministicRecognition({}),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["source-selection-ambiguous"],
            )

    def test_rejects_an_invalid_source_image_before_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_source = root / "missing-source.jpg"
            target_root = root / "Target Person"
            target_root.mkdir()

            outcome = compare(
                ComparisonRequest(source=missing_source, target_root=target_root),
                DeterministicRecognition({}),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.matches, ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["source-image-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, missing_source.resolve())

    def test_rejects_an_unusable_source_image_before_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source image")
            target_root = root / "Target Person"
            target_root.mkdir()

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                FailingRecognition(),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.matches, ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["source-image-unusable"],
            )
            self.assertIn("No usable face", outcome.diagnostics[0].message)
            self.assertEqual(render_matches(outcome), "")

    def test_returns_a_threshold_qualified_target_without_changing_the_face_tree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "arbitrary-source.data"
            source.write_bytes(b"source image")
            target_root = root / "Target Person"
            target_root.mkdir()
            target_image = target_root / "folder.jpg"
            target_image.write_bytes(b"target image")
            existing_cache = target_root / "folder.jpg.facenet512.npy"
            existing_cache.write_bytes(b"existing cache")
            before = snapshot_files(root)

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        target_image: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                outcome.matches,
                (CandidateMatch(identity_path=Path("."), cosine_distance=0.0),),
            )
            self.assertEqual(outcome.diagnostics, ())
            self.assertEqual(outcome.progress, ())
            self.assertEqual(snapshot_files(root), before)

    def test_reports_a_successful_comparison_with_no_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source image")
            target_root = root / "Target Person"
            target_root.mkdir()
            target_image = target_root / "folder.jpg"
            target_image.write_bytes(b"target image")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        target_image: (0.0, 1.0),
                    }
                ),
            )

            self.assertEqual(render_matches(outcome), "No matches found\n")

    def test_presents_a_ranked_candidate_path_and_cosine_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source image")
            target_root = root / "Target Person"
            target_root.mkdir()
            target_image = target_root / "folder.jpg"
            target_image.write_bytes(b"target image")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        target_image: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                render_matches(outcome),
                "Rank  Identity  Cosine distance\n"
                "1     .         0.000000\n",
            )


class FolderSourceComparisonTests(unittest.TestCase):
    def test_requires_the_exact_lowercase_folder_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "Source Person"
            source_folder.mkdir()
            (source_folder / "Folder.JPG").write_bytes(b"lookalike anchor")
            (source_folder / "folder0.jpg").write_bytes(b"numbered image")
            descendant = source_folder / "descendant"
            descendant.mkdir()
            (descendant / "folder.jpg").write_bytes(b"descendant anchor")
            target_root = root / "Target Person"
            target_root.mkdir()

            outcome = compare(
                ComparisonRequest(
                    source_folder=source_folder,
                    target_root=target_root,
                ),
                DeterministicRecognition({}),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["source-folder-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, source_folder.resolve())

    def test_constructs_a_folder_source_from_recognized_root_images_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "Source Person"
            source_folder.mkdir()
            anchor = source_folder / "folder.jpg"
            anchor.write_bytes(b"anchor")
            numbered_folder_image = source_folder / "folder7.JPG"
            numbered_folder_image.write_bytes(b"numbered folder image")
            numbered_face_image = source_folder / "portrait.FACE3.PNG"
            numbered_face_image.write_bytes(b"numbered face image")
            (source_folder / "ordinary.jpg").write_bytes(b"not recognized")
            descendant = source_folder / "descendant"
            descendant.mkdir()
            (descendant / "person.face0.jpg").write_bytes(b"out of scope")

            target_root = root / "Target Person"
            target_root.mkdir()
            target_image = target_root / "folder.jpg"
            target_image.write_bytes(b"target")

            outcome = compare(
                ComparisonRequest(
                    source_folder=source_folder,
                    target_root=target_root,
                ),
                DeterministicRecognition(
                    {
                        anchor: (1.0, 0.0),
                        numbered_folder_image: (3.0, 0.0),
                        numbered_face_image: (0.0, 4.0),
                        target_image: (1.0, 0.0),
                    }
                ),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(len(outcome.matches), 1)
            self.assertAlmostEqual(
                outcome.matches[0].cosine_distance,
                1.0 - 2.0 / (5.0**0.5),
            )
            self.assertEqual(outcome.diagnostics, ())

    def test_warns_for_each_unusable_folder_image_and_uses_the_remainder(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "Source Person"
            source_folder.mkdir()
            anchor = source_folder / "folder.jpg"
            anchor.write_bytes(b"unusable anchor")
            usable_image = source_folder / "folder2.jpg"
            usable_image.write_bytes(b"usable image")
            unusable_image = source_folder / "portrait.face8.webp"
            unusable_image.write_bytes(b"unusable image")

            target_root = root / "Target Person"
            target_root.mkdir()
            target_image = target_root / "folder.jpg"
            target_image.write_bytes(b"target")

            outcome = compare(
                ComparisonRequest(
                    source_folder=source_folder,
                    target_root=target_root,
                ),
                RecognitionWithFailures(
                    vectors={
                        usable_image: (1.0, 0.0),
                        target_image: (1.0, 0.0),
                    },
                    failures={
                        anchor: "Image could not be loaded",
                        unusable_image: "Expected one face but found two",
                    },
                ),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.matches[0].cosine_distance, 0.0)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["source-folder-image-unusable", "source-folder-image-unusable"],
            )
            self.assertEqual(
                {diagnostic.path for diagnostic in outcome.diagnostics},
                {anchor, unusable_image},
            )
            self.assertTrue(
                all(
                    diagnostic.severity == "warning"
                    for diagnostic in outcome.diagnostics
                )
            )

    def test_reports_every_folder_failure_before_the_fatal_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "Source Person"
            source_folder.mkdir()
            anchor = source_folder / "folder.jpg"
            anchor.write_bytes(b"unusable anchor")
            numbered_image = source_folder / "folder1.jpg"
            numbered_image.write_bytes(b"unusable numbered image")
            target_root = root / "Target Person"
            target_root.mkdir()

            outcome = compare(
                ComparisonRequest(
                    source_folder=source_folder,
                    target_root=target_root,
                ),
                RecognitionWithFailures(
                    vectors={},
                    failures={
                        anchor: "No face found",
                        numbered_image: "Image could not be loaded",
                    },
                ),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.matches, ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                [
                    "source-folder-image-unusable",
                    "source-folder-image-unusable",
                    "source-folder-unusable",
                ],
            )
            self.assertEqual(
                [diagnostic.severity for diagnostic in outcome.diagnostics],
                ["warning", "warning", "error"],
            )
            self.assertEqual(outcome.diagnostics[-1].path, source_folder)
            self.assertIn("examined: 2, usable: 0", outcome.diagnostics[-1].message)


class SinglePersonTargetComparisonTests(unittest.TestCase):
    def test_skips_a_target_folder_without_the_exact_lowercase_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Target Person"
            target_root.mkdir()
            (target_root / "Folder.JPG").write_bytes(b"lookalike anchor")
            (target_root / "folder5.JPG").write_bytes(b"numbered folder image")
            (target_root / "portrait.face2.png").write_bytes(b"numbered face")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition({source: (1.0, 0.0)}),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.matches, ())
            self.assertEqual(outcome.diagnostics, ())

    def test_compares_the_normalized_equal_weight_target_folder_vector(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Target Person"
            target_root.mkdir()
            anchor = target_root / "folder.jpg"
            anchor.write_bytes(b"anchor")
            numbered_folder_image = target_root / "folder4.JPG"
            numbered_folder_image.write_bytes(b"numbered folder image")
            numbered_face_image = target_root / "portrait.FACE6.PNG"
            numbered_face_image.write_bytes(b"numbered face image")
            (target_root / "ordinary.jpg").write_bytes(b"not recognized")
            descendant = target_root / "descendant"
            descendant.mkdir()
            (descendant / "folder.jpg").write_bytes(b"out of scope")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        anchor: (1.0, 0.0),
                        numbered_folder_image: (3.0, 0.0),
                        numbered_face_image: (0.0, 4.0),
                    }
                ),
            )

            self.assertEqual(
                outcome.matches[0].identity_path,
                Path("."),
            )
            self.assertAlmostEqual(
                outcome.matches[0].cosine_distance,
                1.0 - 2.0 / (5.0**0.5),
            )
            self.assertEqual(outcome.diagnostics, ())

    def test_warns_for_unusable_target_images_and_keeps_the_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Target Person"
            target_root.mkdir()
            anchor = target_root / "folder.jpg"
            anchor.write_bytes(b"unusable anchor")
            usable_image = target_root / "folder2.jpg"
            usable_image.write_bytes(b"usable image")
            unusable_image = target_root / "portrait.face8.webp"
            unusable_image.write_bytes(b"unusable image")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                RecognitionWithFailures(
                    vectors={
                        source: (1.0, 0.0),
                        usable_image: (3.0, 0.0),
                    },
                    failures={
                        anchor: "Image could not be loaded",
                        unusable_image: "Expected one face but found two",
                    },
                ),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(
                outcome.matches,
                (CandidateMatch(identity_path=Path("."), cosine_distance=0.0),),
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["target-folder-image-unusable", "target-folder-image-unusable"],
            )
            self.assertEqual(
                {diagnostic.path for diagnostic in outcome.diagnostics},
                {anchor, unusable_image},
            )
            self.assertTrue(
                all(
                    diagnostic.severity == "warning"
                    for diagnostic in outcome.diagnostics
                )
            )


if __name__ == "__main__":
    unittest.main()
