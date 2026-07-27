import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

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

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        return self._vectors[image_path]


class FailingRecognition:
    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        raise RecognitionFailure(f"No usable face found in {image_path.name}")


class RecognitionWithFailures:
    def __init__(
        self,
        vectors: dict[Path, tuple[float, ...]],
        failures: dict[Path, str],
    ) -> None:
        self._vectors = vectors
        self._failures = failures

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        if image_path in self._failures:
            raise RecognitionFailure(self._failures[image_path])
        return self._vectors[image_path]


class ProfileSensitiveRecognition:
    def __init__(
        self,
        vectors: dict[tuple[Path, tuple[object, ...]], tuple[float, ...]],
    ) -> None:
        self._vectors = vectors

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        profile_key = (
            profile.model_name,
            profile.cache_slug,
            profile.expected_dimensions,
            profile.cosine_threshold,
            profile.detector_backend,
            profile.align,
        )
        return self._vectors[(image_path, profile_key)]


class RecognitionRemovingDescendant:
    def __init__(
        self,
        vectors: dict[Path, tuple[float, ...]],
        trigger: Path,
        disappearing_face: Path,
    ) -> None:
        self._vectors = vectors
        self._trigger = trigger
        self._disappearing_face = disappearing_face

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        vector = self._vectors[image_path]
        if image_path == self._trigger and self._disappearing_face.exists():
            self._disappearing_face.unlink()
            self._disappearing_face.parent.rmdir()
        return vector


def snapshot_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def unit_vector(index: int = 0) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(512))


class StandaloneComparisonTests(unittest.TestCase):
    def test_uses_the_fixed_facenet512_profile_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source image")
            target_root = root / "Target Person"
            target_root.mkdir()
            target_image = target_root / "folder.jpg"
            target_image.write_bytes(b"target image")
            facenet512_profile = (
                "Facenet512",
                "facenet512",
                512,
                0.30,
                "retinaface",
                True,
            )

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                ProfileSensitiveRecognition(
                    {
                        (source, facenet512_profile): (1.0, 0.0),
                        (target_image, facenet512_profile): (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                outcome.matches,
                (CandidateMatch(identity_path=Path("."), cosine_distance=0.0),),
            )

    def test_selects_arcface_with_its_default_match_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source image")
            target_root = root / "Target Person"
            target_root.mkdir()
            target_image = target_root / "folder.jpg"
            target_image.write_bytes(b"target image")
            arcface_profile = (
                "ArcFace",
                "arcface",
                512,
                0.68,
                "retinaface",
                True,
            )

            outcome = compare(
                ComparisonRequest(
                    source=source,
                    target_root=target_root,
                    model_name="ArcFace",
                ),
                ProfileSensitiveRecognition(
                    {
                        (source, arcface_profile): (1.0, 0.0),
                        (target_image, arcface_profile): (0.5, 0.8660254037844386),
                    }
                ),
            )

            self.assertEqual(len(outcome.matches), 1)
            self.assertAlmostEqual(outcome.matches[0].cosine_distance, 0.5)

    def test_rejects_an_unsupported_recognition_model_before_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source image")
            target_root = root / "Target Person"
            target_root.mkdir()
            (target_root / "folder.jpg").write_bytes(b"target image")

            outcome = compare(
                ComparisonRequest(
                    source=source,
                    target_root=target_root,
                    model_name="VGG-Face",
                ),
                DeterministicRecognition({}),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.matches, ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["recognition-model-unsupported"],
            )

    def test_rejects_an_invalid_threshold_before_target_work(self) -> None:
        invalid_thresholds = (
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.01,
            2.01,
        )
        for invalid_threshold in invalid_thresholds:
            with self.subTest(threshold=invalid_threshold):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "source.jpg"
                    source.write_bytes(b"source image")
                    target_root = root / "Target Person"
                    target_root.mkdir()
                    (target_root / "folder.jpg").write_bytes(b"target image")

                    outcome = compare(
                        ComparisonRequest(
                            source=source,
                            target_root=target_root,
                            threshold=invalid_threshold,
                        ),
                        DeterministicRecognition({}),
                    )

                self.assertFalse(outcome.successful)
                self.assertEqual(outcome.matches, ())
                self.assertEqual(
                    [diagnostic.code for diagnostic in outcome.diagnostics],
                    ["match-threshold-invalid"],
                )

    def test_accepts_inclusive_threshold_override_boundaries(self) -> None:
        cases = (
            (0.0, (1.0, 0.0), 0.0),
            (2.0, (-1.0, 0.0), 2.0),
        )
        for threshold, target_vector, expected_distance in cases:
            with self.subTest(threshold=threshold):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "source.jpg"
                    source.write_bytes(b"source image")
                    target_root = root / "Target Person"
                    target_root.mkdir()
                    target_image = target_root / "folder.jpg"
                    target_image.write_bytes(b"target image")

                    outcome = compare(
                        ComparisonRequest(
                            source=source,
                            target_root=target_root,
                            threshold=threshold,
                        ),
                        DeterministicRecognition(
                            {
                                source: (1.0, 0.0),
                                target_image: target_vector,
                            }
                        ),
                    )

                self.assertEqual(len(outcome.matches), 1)
                self.assertAlmostEqual(
                    outcome.matches[0].cosine_distance,
                    expected_distance,
                )

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
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["target-cache-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, existing_cache)
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
            descendant_anchor = descendant / "folder.jpg"
            descendant_anchor.write_bytes(b"separate descendant target")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        anchor: (1.0, 0.0),
                        numbered_folder_image: (3.0, 0.0),
                        numbered_face_image: (0.0, 4.0),
                        descendant_anchor: (0.0, 1.0),
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


class MultiPersonTargetComparisonTests(unittest.TestCase):
    def test_orders_threshold_qualified_candidates_by_cosine_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Event Photos"
            target_root.mkdir()
            farther_face = target_root / "A.face0.jpg"
            farther_face.write_bytes(b"farther face")
            nearer_face = target_root / "Z.face1.jpg"
            nearer_face.write_bytes(b"nearer face")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        farther_face: (0.8, 0.6),
                        nearer_face: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                [match.identity_path for match in outcome.matches],
                [Path("Z.face1.jpg"), Path("A.face0.jpg")],
            )

    def test_compares_sparse_numbered_faces_as_independent_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Event Photos"
            target_root.mkdir()
            first_face = target_root / "Alice.FACE9.JPG"
            first_face.write_bytes(b"first face")
            second_face = target_root / "Bob.face2.PNG"
            second_face.write_bytes(b"second face")
            (target_root / "folder3.jpg").write_bytes(b"numbered folder image")
            (target_root / "ordinary.jpg").write_bytes(b"ordinary image")
            (target_root / "orphan.facenet512.npy").write_bytes(b"cache")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        first_face: (1.0, 0.0),
                        second_face: (0.8, 0.6),
                    }
                ),
            )

            self.assertEqual(
                outcome.matches,
                (
                    CandidateMatch(
                        identity_path=Path("Alice.FACE9.JPG"),
                        cosine_distance=0.0,
                    ),
                    CandidateMatch(
                        identity_path=Path("Bob.face2.PNG"),
                        cosine_distance=0.19999999999999996,
                    ),
                ),
            )
            self.assertEqual(outcome.diagnostics, ())

    def test_warns_for_an_unusable_numbered_face_and_keeps_the_others(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Event Photos"
            target_root.mkdir()
            unusable_face = target_root / "group.face1.jpg"
            unusable_face.write_bytes(b"multiple faces")
            usable_face = target_root / "portrait.face7.jpeg"
            usable_face.write_bytes(b"one face")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                RecognitionWithFailures(
                    vectors={
                        source: (1.0, 0.0),
                        usable_face: (1.0, 0.0),
                    },
                    failures={
                        unusable_face: "Expected one face but found two",
                    },
                ),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(
                outcome.matches,
                (
                    CandidateMatch(
                        identity_path=Path("portrait.face7.jpeg"),
                        cosine_distance=0.0,
                    ),
                ),
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["target-face-unusable"],
            )
            self.assertEqual(outcome.diagnostics[0].path, unusable_face)
            self.assertEqual(outcome.diagnostics[0].severity, "warning")

    def test_compares_static_webp_and_excludes_animated_webp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Event Photos"
            target_root.mkdir()
            static_webp = target_root / "Still.FACE4.WeBp"
            Image.new("RGB", (2, 2), "red").save(static_webp, format="WEBP")
            animated_webp = target_root / "Motion.face8.webp"
            frames = [
                Image.new("RGB", (2, 2), "red"),
                Image.new("RGB", (2, 2), "blue"),
            ]
            frames[0].save(
                animated_webp,
                format="WEBP",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        static_webp: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                outcome.matches,
                (
                    CandidateMatch(
                        identity_path=Path("Still.FACE4.WeBp"),
                        cosine_distance=0.0,
                    ),
                ),
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["animated-webp-unsupported"],
            )
            self.assertEqual(outcome.diagnostics[0].path, animated_webp)
            self.assertEqual(outcome.diagnostics[0].severity, "warning")


class LiveFaceTreeTraversalTests(unittest.TestCase):
    def test_rejects_a_missing_or_non_directory_target_root_before_recognition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            missing_root = root / "Missing"
            file_root = root / "not-a-directory"
            file_root.write_bytes(b"file")

            for invalid_root in (missing_root, file_root):
                with self.subTest(target_root=invalid_root):
                    outcome = compare(
                        ComparisonRequest(source=source, target_root=invalid_root),
                        DeterministicRecognition({}),
                    )

                self.assertFalse(outcome.successful)
                self.assertEqual(outcome.matches, ())
                self.assertEqual(
                    [diagnostic.code for diagnostic in outcome.diagnostics],
                    ["target-root-invalid"],
                )
                self.assertEqual(outcome.diagnostics[0].path, invalid_root.resolve())

    def test_rejects_an_unreadable_selected_target_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Unreadable Face Tree"
            target_root.mkdir()
            (target_root / "Person.face0.jpg").write_bytes(b"hidden face")
            target_root.chmod(0)

            try:
                outcome = compare(
                    ComparisonRequest(source=source, target_root=target_root),
                    DeterministicRecognition({}),
                )
            finally:
                target_root.chmod(0o700)

            self.assertFalse(outcome.successful)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["target-root-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, target_root)

    def test_compares_the_selected_root_and_descendants_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Face Tree"
            target_root.mkdir()
            root_face = target_root / "Root.face0.jpg"
            root_face.write_bytes(b"root face")
            upper_branch = target_root / "Album"
            upper_branch.mkdir()
            upper_face = upper_branch / "Shared.face1.jpg"
            upper_face.write_bytes(b"upper face")
            lower_branch = target_root / "album"
            lower_branch.mkdir()
            lower_face = lower_branch / "Shared.face1.jpg"
            lower_face.write_bytes(b"lower face")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        root_face: (1.0, 0.0),
                        upper_face: (1.0, 0.0),
                        lower_face: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                {match.identity_path for match in outcome.matches},
                {
                    Path("Root.face0.jpg"),
                    Path("Album/Shared.face1.jpg"),
                    Path("album/Shared.face1.jpg"),
                },
            )

    def test_single_target_folder_mode_excludes_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Face Tree"
            target_root.mkdir()
            root_face = target_root / "Root.face0.jpg"
            root_face.write_bytes(b"root face")
            descendant = target_root / "Descendant"
            descendant.mkdir()
            descendant_face = descendant / "Child.face1.jpg"
            descendant_face.write_bytes(b"descendant face")

            outcome = compare(
                ComparisonRequest(
                    source=source,
                    target_root=target_root,
                    single_target_folder=True,
                ),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        root_face: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                [match.identity_path for match in outcome.matches],
                [Path("Root.face0.jpg")],
            )

    def test_warns_about_symlinked_descendants_without_following_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Face Tree"
            target_root.mkdir()
            usable_face = target_root / "Usable.face2.jpg"
            usable_face.write_bytes(b"usable face")

            external = root / "External"
            external.mkdir()
            external_face = external / "Hidden.face0.jpg"
            external_face.write_bytes(b"external face")
            symlinked_directory = target_root / "Linked Album"
            symlinked_directory.symlink_to(external, target_is_directory=True)
            symlinked_face = target_root / "Linked.face1.jpg"
            symlinked_face.symlink_to(external_face)
            external_cache = root / "external-cache.npy"
            external_cache.write_bytes(b"cache")
            symlinked_cache = target_root / "Usable.face2.jpg.facenet512.npy"
            symlinked_cache.symlink_to(external_cache)

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        usable_face: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                [match.identity_path for match in outcome.matches],
                [Path("Usable.face2.jpg")],
            )
            self.assertEqual(
                {diagnostic.path for diagnostic in outcome.diagnostics},
                {symlinked_directory, symlinked_face, symlinked_cache},
            )
            self.assertTrue(
                all(
                    diagnostic.code == "target-symlink-skipped"
                    for diagnostic in outcome.diagnostics
                )
            )

    def test_excludes_the_single_person_identity_containing_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_root = Path(temporary_directory) / "Face Tree"
            target_root.mkdir()
            source_identity = target_root / "Source Person"
            source_identity.mkdir()
            source_anchor = source_identity / "folder.jpg"
            source_anchor.write_bytes(b"anchor")
            source = source_identity / "folder1.jpg"
            source.write_bytes(b"selected source")
            other_identity = target_root / "Other Person"
            other_identity.mkdir()
            other_anchor = other_identity / "folder.jpg"
            other_anchor.write_bytes(b"other anchor")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        source_anchor: (1.0, 0.0),
                        other_anchor: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                [match.identity_path for match in outcome.matches],
                [Path("Other Person")],
            )

    def test_excludes_only_the_selected_multi_person_face(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_root = Path(temporary_directory) / "Event Photos"
            target_root.mkdir()
            source = target_root / "Selected.face0.jpg"
            source.write_bytes(b"selected source")
            other_face = target_root / "Other.face1.jpg"
            other_face.write_bytes(b"other face")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: (1.0, 0.0),
                        other_face: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                [match.identity_path for match in outcome.matches],
                [Path("Other.face1.jpg")],
            )

    def test_excludes_a_source_folder_that_is_inside_the_target_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_root = Path(temporary_directory) / "Face Tree"
            target_root.mkdir()
            source_folder = target_root / "Source Person"
            source_folder.mkdir()
            source_anchor = source_folder / "folder.jpg"
            source_anchor.write_bytes(b"source anchor")
            other_identity = target_root / "Other Person"
            other_identity.mkdir()
            other_anchor = other_identity / "folder.jpg"
            other_anchor.write_bytes(b"other anchor")

            outcome = compare(
                ComparisonRequest(
                    source_folder=source_folder,
                    target_root=target_root,
                ),
                DeterministicRecognition(
                    {
                        source_anchor: (1.0, 0.0),
                        other_anchor: (1.0, 0.0),
                    }
                ),
            )

            self.assertEqual(
                [match.identity_path for match in outcome.matches],
                [Path("Other Person")],
            )

    def test_warns_when_a_discovered_descendant_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Face Tree"
            target_root.mkdir()
            usable_directory = target_root / "A Usable"
            usable_directory.mkdir()
            usable_face = usable_directory / "Person.face0.jpg"
            usable_face.write_bytes(b"usable face")
            disappearing_directory = target_root / "Z Disappearing"
            disappearing_directory.mkdir()
            disappearing_face = disappearing_directory / "Person.face0.jpg"
            disappearing_face.write_bytes(b"disappearing face")

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                RecognitionRemovingDescendant(
                    vectors={
                        source: (1.0, 0.0),
                        usable_face: (1.0, 0.0),
                    },
                    trigger=usable_face,
                    disappearing_face=disappearing_face,
                ),
            )

            self.assertEqual(
                [match.identity_path for match in outcome.matches],
                [Path("A Usable/Person.face0.jpg")],
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["target-descendant-unavailable"],
            )
            self.assertEqual(outcome.diagnostics[0].path, disappearing_directory)

    def test_warns_for_an_unreadable_descendant_and_keeps_other_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Face Tree"
            target_root.mkdir()
            usable_directory = target_root / "A Usable"
            usable_directory.mkdir()
            usable_face = usable_directory / "Person.face0.jpg"
            usable_face.write_bytes(b"usable face")
            unreadable_directory = target_root / "Z Unreadable"
            unreadable_directory.mkdir()
            (unreadable_directory / "Person.face0.jpg").write_bytes(b"hidden face")
            unreadable_directory.chmod(0)

            try:
                outcome = compare(
                    ComparisonRequest(source=source, target_root=target_root),
                    DeterministicRecognition(
                        {
                            source: (1.0, 0.0),
                            usable_face: (1.0, 0.0),
                        }
                    ),
                )
            finally:
                unreadable_directory.chmod(0o700)

            self.assertEqual(
                [match.identity_path for match in outcome.matches],
                [Path("A Usable/Person.face0.jpg")],
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["target-descendant-unavailable"],
            )
            self.assertEqual(outcome.diagnostics[0].path, unreadable_directory)


class VectorCacheReuseTests(unittest.TestCase):
    def test_reuses_a_compatible_multi_person_target_cache_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Event Photos"
            target_root.mkdir()
            target_face = target_root / "Person.face0.jpg"
            target_face.write_bytes(b"target face")
            cache = target_root / "Person.face0.jpg.facenet512.npy"
            np.save(cache, np.asarray(unit_vector()))
            before = snapshot_files(root)

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition({source: unit_vector()}),
            )

            self.assertEqual(
                outcome.matches,
                (
                    CandidateMatch(
                        identity_path=Path("Person.face0.jpg"),
                        cosine_distance=0.0,
                    ),
                ),
            )
            self.assertEqual(outcome.diagnostics, ())
            self.assertEqual(snapshot_files(root), before)

    def test_reuses_folder_aggregate_caches_associated_with_the_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "Source Person"
            source_folder.mkdir()
            source_anchor = source_folder / "folder.jpg"
            source_anchor.write_bytes(b"source anchor")
            np.save(
                source_folder / "folder.jpg.facenet512.npy",
                np.asarray(unit_vector()),
            )
            target_root = root / "Target Person"
            target_root.mkdir()
            target_anchor = target_root / "folder.jpg"
            target_anchor.write_bytes(b"target anchor")
            np.save(
                target_root / "folder.jpg.facenet512.npy",
                np.asarray(unit_vector()),
            )
            before = snapshot_files(root)

            outcome = compare(
                ComparisonRequest(
                    source_folder=source_folder,
                    target_root=target_root,
                ),
                DeterministicRecognition({}),
            )

            self.assertEqual(
                outcome.matches,
                (CandidateMatch(identity_path=Path("."), cosine_distance=0.0),),
            )
            self.assertEqual(outcome.diagnostics, ())
            self.assertEqual(snapshot_files(root), before)

    def test_can_disable_cache_reuse_for_a_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "Source Person"
            source_folder.mkdir()
            source_anchor = source_folder / "folder.jpg"
            source_anchor.write_bytes(b"source anchor")
            np.save(
                source_folder / "folder.jpg.facenet512.npy",
                np.asarray(unit_vector(1)),
            )
            target_root = root / "Target Person"
            target_root.mkdir()
            target_anchor = target_root / "folder.jpg"
            target_anchor.write_bytes(b"target anchor")
            np.save(
                target_root / "folder.jpg.facenet512.npy",
                np.asarray(unit_vector(2)),
            )

            outcome = compare(
                ComparisonRequest(
                    source_folder=source_folder,
                    target_root=target_root,
                    reuse_cache=False,
                ),
                DeterministicRecognition(
                    {
                        source_anchor: unit_vector(),
                        target_anchor: unit_vector(),
                    }
                ),
            )

            self.assertEqual(
                outcome.matches,
                (CandidateMatch(identity_path=Path("."), cosine_distance=0.0),),
            )
            self.assertEqual(outcome.diagnostics, ())

    def test_never_reuses_a_cache_beside_a_standalone_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            np.save(
                root / "source.jpg.facenet512.npy",
                np.asarray(unit_vector(1)),
            )
            target_root = root / "Event Photos"
            target_root.mkdir()
            target_face = target_root / "Person.face0.jpg"
            target_face.write_bytes(b"target face")
            np.save(
                target_root / "Person.face0.jpg.facenet512.npy",
                np.asarray(unit_vector()),
            )

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition({source: unit_vector()}),
            )

            self.assertEqual(len(outcome.matches), 1)

    def test_warns_and_recalculates_an_incompatible_target_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Event Photos"
            target_root.mkdir()
            target_face = target_root / "Person.face0.jpg"
            target_face.write_bytes(b"target face")
            invalid_cache = target_root / "Person.face0.jpg.facenet512.npy"
            np.save(invalid_cache, np.asarray((1.0, 0.0)))
            before = snapshot_files(root)

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: unit_vector(),
                        target_face: unit_vector(),
                    }
                ),
            )

            self.assertEqual(len(outcome.matches), 1)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["target-cache-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, invalid_cache)
            self.assertEqual(snapshot_files(root), before)

    def test_warns_and_recalculates_an_invalid_source_folder_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_folder = root / "Source Person"
            source_folder.mkdir()
            source_anchor = source_folder / "folder.jpg"
            source_anchor.write_bytes(b"source anchor")
            invalid_cache = source_folder / "folder.jpg.facenet512.npy"
            invalid_cache.write_bytes(b"not an npy file")
            target_root = root / "Target Person"
            target_root.mkdir()
            target_anchor = target_root / "folder.jpg"
            target_anchor.write_bytes(b"target anchor")
            np.save(
                target_root / "folder.jpg.facenet512.npy",
                np.asarray(unit_vector()),
            )
            before = snapshot_files(root)

            outcome = compare(
                ComparisonRequest(
                    source_folder=source_folder,
                    target_root=target_root,
                ),
                DeterministicRecognition({source_anchor: unit_vector()}),
            )

            self.assertEqual(len(outcome.matches), 1)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["source-cache-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, invalid_cache)
            self.assertEqual(snapshot_files(root), before)

    def test_warns_and_recalculates_an_invalid_target_folder_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Target Person"
            target_root.mkdir()
            target_anchor = target_root / "folder.jpg"
            target_anchor.write_bytes(b"target anchor")
            invalid_cache = target_root / "folder.jpg.facenet512.npy"
            np.save(invalid_cache, np.asarray(["not numeric"] * 512))
            before = snapshot_files(root)

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: unit_vector(),
                        target_anchor: unit_vector(),
                    }
                ),
            )

            self.assertEqual(len(outcome.matches), 1)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["target-cache-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, invalid_cache)
            self.assertEqual(snapshot_files(root), before)

    def test_uses_only_the_exact_case_sensitive_selected_model_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Event Photos"
            target_root.mkdir()
            target_face = target_root / "Person.face0.jpg"
            target_face.write_bytes(b"target face")
            np.save(
                target_root / "Person.face0.jpg.arcface.npy",
                np.asarray(unit_vector()),
            )
            np.save(
                target_root / "Person.face0.jpg.ArcFace.npy",
                np.asarray(unit_vector(1)),
            )
            np.save(
                target_root / "Person.face0.jpg.facenet512.npy",
                np.asarray(unit_vector(2)),
            )
            before = snapshot_files(root)

            outcome = compare(
                ComparisonRequest(
                    source=source,
                    target_root=target_root,
                    model_name="ArcFace",
                ),
                DeterministicRecognition({source: unit_vector()}),
            )

            self.assertEqual(len(outcome.matches), 1)
            self.assertEqual(outcome.diagnostics, ())
            self.assertEqual(snapshot_files(root), before)

    def test_compares_a_hybrid_cached_and_transient_target_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "Event Photos"
            target_root.mkdir()
            cached_face = target_root / "Cached.face0.jpg"
            cached_face.write_bytes(b"cached target")
            np.save(
                target_root / "Cached.face0.jpg.facenet512.npy",
                np.asarray(unit_vector()),
            )
            transient_face = target_root / "Transient.face1.jpg"
            transient_face.write_bytes(b"transient target")
            before = snapshot_files(root)

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                DeterministicRecognition(
                    {
                        source: unit_vector(),
                        transient_face: unit_vector(),
                    }
                ),
            )

            self.assertEqual(
                {match.identity_path for match in outcome.matches},
                {Path("Cached.face0.jpg"), Path("Transient.face1.jpg")},
            )
            self.assertEqual(outcome.diagnostics, ())
            self.assertEqual(snapshot_files(root), before)


if __name__ == "__main__":
    unittest.main()
