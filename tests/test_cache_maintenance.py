import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from faceledger.comparison import AssetAcquisitionFailure, RecognitionFailure
from faceledger.maintenance import (
    CacheBuildRequest,
    build_vector_cache,
    rebuild_vector_cache,
)


class DeepFaceModule(types.ModuleType):
    DeepFace: object


def unit_vector(index: int = 0) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(512))


class DeterministicRecognition:
    def __init__(self, vectors: dict[Path, tuple[float, ...]]) -> None:
        self._vectors = vectors

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        return self._vectors[image_path]


class RecognitionWithFailures(DeterministicRecognition):
    def __init__(
        self,
        vectors: dict[Path, tuple[float, ...]],
        failures: dict[Path, str],
    ) -> None:
        super().__init__(vectors)
        self._failures = failures

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        if image_path in self._failures:
            raise RecognitionFailure(self._failures[image_path])
        return super().vector_for(image_path, profile)


class RecognitionWithAssetFailure:
    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        raise AssetAcquisitionFailure("Model assets unavailable; retry online.")


class CacheBuildTests(unittest.TestCase):
    def test_public_build_uses_the_locked_deepface_adapter_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            deepface_home = temporary_path / "deepface-home"
            weights = deepface_home / ".deepface" / "weights"
            weights.mkdir(parents=True)
            (weights / "facenet512_weights.h5").write_bytes(b"model")
            (weights / "retinaface.h5").write_bytes(b"detector")
            deepface_module = DeepFaceModule("deepface")
            deepface_module.DeepFace = types.SimpleNamespace(
                represent=lambda **_arguments: [{"embedding": unit_vector()}]
            )

            with (
                patch.dict(os.environ, {"DEEPFACE_HOME": str(deepface_home)}),
                patch.dict(sys.modules, {"deepface": deepface_module}),
            ):
                outcome = build_vector_cache(CacheBuildRequest(root=root))

            cache = root / "Person.face0.jpg.facenet512.npy"
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.created, (cache,))
            np.testing.assert_array_equal(np.load(cache), unit_vector())

    def test_rejects_an_unreadable_selected_maintenance_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Unreadable Face Tree"
            root.mkdir()
            (root / "Person.face0.jpg").write_bytes(b"face")
            root.chmod(0)

            try:
                outcome = build_vector_cache(
                    CacheBuildRequest(root=root),
                    DeterministicRecognition({}),
                )
            finally:
                root.chmod(0o700)

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.created, ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["maintenance-root-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, root)

    def test_skips_a_symlinked_cache_entry_without_replacing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            outside_cache = temporary_path / "outside.npy"
            outside_cache.write_bytes(b"outside cache")
            cache = root / "Person.face0.jpg.facenet512.npy"
            cache.symlink_to(outside_cache)

            outcome = build_vector_cache(
                CacheBuildRequest(root=root),
                DeterministicRecognition({}),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.created, ())
            self.assertTrue(cache.is_symlink())
            self.assertEqual(outside_cache.read_bytes(), b"outside cache")
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["maintenance-symlink-skipped"],
            )
            self.assertEqual(outcome.diagnostics[0].path, cache)

    def test_builds_static_webp_and_skips_animated_webp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            static_webp = root / "Still.face0.webp"
            Image.new("RGB", (2, 2), "red").save(static_webp, format="WEBP")
            animated_webp = root / "Motion.face1.webp"
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

            outcome = build_vector_cache(
                CacheBuildRequest(root=root),
                DeterministicRecognition({static_webp: unit_vector()}),
            )

            static_cache = root / "Still.face0.webp.facenet512.npy"
            animated_cache = root / "Motion.face1.webp.facenet512.npy"
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.created, (static_cache,))
            self.assertFalse(animated_cache.exists())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["animated-webp-unsupported"],
            )
            self.assertEqual(outcome.diagnostics[0].path, animated_webp)

    def test_builds_only_the_selected_model_in_the_selected_root_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            root_face = root / "Root.face0.jpg"
            root_face.write_bytes(b"root face")
            other_model_cache = root / "Root.face0.jpg.arcface.npy"
            other_model_cache.write_bytes(b"existing ArcFace cache")
            descendant = root / "Descendant"
            descendant.mkdir()
            descendant_face = descendant / "Nested.face1.jpg"
            descendant_face.write_bytes(b"descendant face")

            outcome = build_vector_cache(
                CacheBuildRequest(root=root, model_name="Facenet512"),
                DeterministicRecognition({root_face: unit_vector()}),
            )

            created_cache = root / "Root.face0.jpg.facenet512.npy"
            descendant_cache = descendant / "Nested.face1.jpg.facenet512.npy"
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.created, (created_cache,))
            np.testing.assert_array_equal(np.load(created_cache), unit_vector())
            self.assertFalse(descendant_cache.exists())
            self.assertEqual(
                other_model_cache.read_bytes(),
                b"existing ArcFace cache",
            )

    def test_recursive_build_retains_compatible_entries_and_adds_descendants(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            root_face = root / "Root.face0.jpg"
            root_face.write_bytes(b"root face")
            compatible_cache = root / "Root.face0.jpg.facenet512.npy"
            np.save(compatible_cache, np.asarray(unit_vector(0)))
            before = compatible_cache.read_bytes()
            descendant = root / "Descendant"
            descendant.mkdir()
            descendant_face = descendant / "Nested.face1.jpg"
            descendant_face.write_bytes(b"descendant face")

            outcome = build_vector_cache(
                CacheBuildRequest(
                    root=root,
                    model_name="Facenet512",
                    recursive=True,
                ),
                DeterministicRecognition({descendant_face: unit_vector(1)}),
            )

            descendant_cache = descendant / "Nested.face1.jpg.facenet512.npy"
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.retained, (compatible_cache,))
            self.assertEqual(outcome.created, (descendant_cache,))
            self.assertEqual(compatible_cache.read_bytes(), before)
            np.testing.assert_array_equal(
                np.load(descendant_cache),
                unit_vector(1),
            )

    def test_warns_and_replaces_every_structurally_invalid_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            faces = tuple(root / f"Person.face{number}.jpg" for number in range(3))
            for face in faces:
                face.write_bytes(b"face")
            caches = tuple(
                root / f"Person.face{number}.jpg.facenet512.npy" for number in range(3)
            )
            caches[0].write_bytes(b"not an NPY file")
            np.save(caches[1], np.asarray(["not numeric"] * 512))
            np.save(caches[2], np.asarray([1.0, 0.0]))

            outcome = build_vector_cache(
                CacheBuildRequest(root=root),
                DeterministicRecognition(
                    {face: unit_vector(number) for number, face in enumerate(faces)}
                ),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.created, caches)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["cache-entry-invalid"] * 3,
            )
            self.assertEqual(
                {diagnostic.path for diagnostic in outcome.diagnostics},
                set(caches),
            )
            self.assertTrue(
                all(
                    diagnostic.severity == "warning"
                    for diagnostic in outcome.diagnostics
                )
            )
            for number, cache in enumerate(caches):
                np.testing.assert_array_equal(np.load(cache), unit_vector(number))

    def test_builds_one_folder_cache_from_the_remaining_usable_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Single Person"
            root.mkdir()
            anchor = root / "folder.jpg"
            anchor.write_bytes(b"unusable anchor")
            numbered = root / "folder2.jpg"
            numbered.write_bytes(b"usable numbered image")
            face = root / "Portrait.face8.webp"
            face.write_bytes(b"usable face image")

            outcome = build_vector_cache(
                CacheBuildRequest(root=root),
                RecognitionWithFailures(
                    vectors={
                        numbered: unit_vector(0),
                        face: unit_vector(1),
                    },
                    failures={anchor: "No face found"},
                ),
            )

            folder_cache = root / "folder.jpg.facenet512.npy"
            expected = np.asarray(
                tuple(2**-0.5 if position in {0, 1} else 0.0 for position in range(512))
            )
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.created, (folder_cache,))
            np.testing.assert_allclose(np.load(folder_cache), expected)
            self.assertFalse((root / "folder2.jpg.facenet512.npy").exists())
            self.assertFalse((root / "Portrait.face8.webp.facenet512.npy").exists())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["cache-build-image-unusable"],
            )
            self.assertEqual(outcome.diagnostics[0].path, anchor)

    def test_warns_for_item_failures_and_continues_building(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            unusable_face = root / "A.face0.jpg"
            unusable_face.write_bytes(b"unusable")
            unwritable_face = root / "B.face1.jpg"
            unwritable_face.write_bytes(b"usable but cache path is blocked")
            blocked_cache = root / "B.face1.jpg.facenet512.npy"
            blocked_cache.mkdir()
            usable_face = root / "C.face2.jpg"
            usable_face.write_bytes(b"usable")

            outcome = build_vector_cache(
                CacheBuildRequest(root=root),
                RecognitionWithFailures(
                    vectors={
                        unwritable_face: unit_vector(1),
                        usable_face: unit_vector(2),
                    },
                    failures={unusable_face: "Expected one face but found two"},
                ),
            )

            usable_cache = root / "C.face2.jpg.facenet512.npy"
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.created, (usable_cache,))
            np.testing.assert_array_equal(np.load(usable_cache), unit_vector(2))
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["cache-build-image-unusable", "cache-build-write-failed"],
            )
            self.assertEqual(outcome.diagnostics[0].path, unusable_face)
            self.assertEqual(outcome.diagnostics[1].path, blocked_cache)
            self.assertTrue(
                all(
                    diagnostic.severity == "warning"
                    for diagnostic in outcome.diagnostics
                )
            )


class CacheRebuildTests(unittest.TestCase):
    def test_public_rebuild_uses_the_locked_deepface_adapter_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            cache = root / "Person.face0.jpg.facenet512.npy"
            np.save(cache, np.asarray(unit_vector(1)))
            deepface_home = temporary_path / "deepface-home"
            weights = deepface_home / ".deepface" / "weights"
            weights.mkdir(parents=True)
            (weights / "facenet512_weights.h5").write_bytes(b"model")
            (weights / "retinaface.h5").write_bytes(b"detector")
            deepface_module = DeepFaceModule("deepface")
            deepface_module.DeepFace = types.SimpleNamespace(
                represent=lambda **_arguments: [{"embedding": unit_vector(2)}]
            )

            with (
                patch.dict(os.environ, {"DEEPFACE_HOME": str(deepface_home)}),
                patch.dict(sys.modules, {"deepface": deepface_module}),
            ):
                outcome = rebuild_vector_cache(CacheBuildRequest(root=root))

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.rebuilt, (cache,))
            np.testing.assert_array_equal(np.load(cache), unit_vector(2))

    def test_asset_acquisition_failure_is_an_operation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            cache = root / "Person.face0.jpg.facenet512.npy"
            np.save(cache, np.asarray(unit_vector(0)))
            old_bytes = cache.read_bytes()

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root),
                RecognitionWithAssetFailure(),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.rebuilt, ())
            self.assertEqual(cache.read_bytes(), old_bytes)
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["model-assets-unavailable"],
            )
            self.assertEqual(outcome.diagnostics[0].severity, "error")

    def test_rejects_an_unsupported_model_before_rebuild_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root, model_name="Unsupported"),
                DeterministicRecognition({}),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.rebuilt, ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["recognition-model-unsupported"],
            )

    def test_rejects_an_invalid_maintenance_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Missing Face Tree"

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root),
                DeterministicRecognition({}),
            )

            self.assertFalse(outcome.successful)
            self.assertEqual(outcome.rebuilt, ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["maintenance-root-invalid"],
            )
            self.assertEqual(outcome.diagnostics[0].path, root)

    def test_recalculates_and_replaces_a_compatible_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            face = root / "Person.face0.jpg"
            face.write_bytes(b"face")
            cache = root / "Person.face0.jpg.facenet512.npy"
            np.save(cache, np.asarray(unit_vector(0)))

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root),
                DeterministicRecognition({face: unit_vector(1)}),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.rebuilt, (cache,))
            self.assertEqual(outcome.diagnostics, ())
            np.testing.assert_array_equal(np.load(cache), unit_vector(1))

    def test_failed_recalculation_preserves_the_old_entry_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            failed_face = root / "Alice.face0.jpg"
            failed_face.write_bytes(b"failed face")
            failed_cache = root / "Alice.face0.jpg.facenet512.npy"
            np.save(failed_cache, np.asarray(unit_vector(0)))
            old_bytes = failed_cache.read_bytes()
            rebuilt_face = root / "Bob.face1.jpg"
            rebuilt_face.write_bytes(b"rebuilt face")
            rebuilt_cache = root / "Bob.face1.jpg.facenet512.npy"
            np.save(rebuilt_cache, np.asarray(unit_vector(0)))

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root),
                RecognitionWithFailures(
                    vectors={rebuilt_face: unit_vector(1)},
                    failures={failed_face: "Expected one face but found two"},
                ),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.rebuilt, (rebuilt_cache,))
            self.assertEqual(failed_cache.read_bytes(), old_bytes)
            np.testing.assert_array_equal(np.load(rebuilt_cache), unit_vector(1))
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["cache-rebuild-image-unusable"],
            )
            self.assertEqual(outcome.diagnostics[0].path, failed_face)
            self.assertEqual(outcome.diagnostics[0].severity, "warning")

    def test_failed_persistence_preserves_the_old_entry_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            failed_face = root / "Alice.face0.jpg"
            failed_face.write_bytes(b"first")
            failed_cache = root / "Alice.face0.jpg.facenet512.npy"
            np.save(failed_cache, np.asarray(unit_vector(0)))
            old_bytes = failed_cache.read_bytes()
            rebuilt_face = root / "Bob.face1.jpg"
            rebuilt_face.write_bytes(b"second")
            rebuilt_cache = root / "Bob.face1.jpg.facenet512.npy"
            np.save(rebuilt_cache, np.asarray(unit_vector(0)))
            real_replace = os.replace

            def replace_unless_first(source: Path, destination: Path) -> None:
                if Path(destination) == failed_cache:
                    raise OSError("replacement denied")
                real_replace(source, destination)

            with patch(
                "faceledger.maintenance.os.replace",
                side_effect=replace_unless_first,
            ):
                outcome = rebuild_vector_cache(
                    CacheBuildRequest(root=root),
                    DeterministicRecognition(
                        {
                            failed_face: unit_vector(1),
                            rebuilt_face: unit_vector(2),
                        }
                    ),
                )

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.rebuilt, (rebuilt_cache,))
            self.assertEqual(failed_cache.read_bytes(), old_bytes)
            np.testing.assert_array_equal(np.load(rebuilt_cache), unit_vector(2))
            self.assertEqual(
                tuple(root.glob(f".{failed_cache.name}.*.tmp")),
                (),
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["cache-rebuild-write-failed"],
            )
            self.assertEqual(outcome.diagnostics[0].path, failed_cache)
            self.assertEqual(outcome.diagnostics[0].severity, "warning")

    def test_rebuilds_one_folder_identity_from_its_remaining_usable_images(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Single Person"
            root.mkdir()
            anchor = root / "folder.jpg"
            anchor.write_bytes(b"unusable anchor")
            numbered = root / "folder2.jpg"
            numbered.write_bytes(b"usable numbered image")
            face = root / "Portrait.face8.webp"
            face.write_bytes(b"usable face image")
            folder_cache = root / "folder.jpg.facenet512.npy"
            np.save(folder_cache, np.asarray(unit_vector(2)))
            old_numbered_cache = root / "folder2.jpg.facenet512.npy"
            old_numbered_cache.write_bytes(b"unrelated per-image file")

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root),
                RecognitionWithFailures(
                    vectors={
                        numbered: unit_vector(0),
                        face: unit_vector(1),
                    },
                    failures={anchor: "No face found"},
                ),
            )

            expected = np.asarray(
                tuple(2**-0.5 if position in {0, 1} else 0.0 for position in range(512))
            )
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.rebuilt, (folder_cache,))
            np.testing.assert_allclose(np.load(folder_cache), expected)
            self.assertEqual(
                old_numbered_cache.read_bytes(),
                b"unrelated per-image file",
            )
            self.assertFalse((root / "Portrait.face8.webp.facenet512.npy").exists())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["cache-rebuild-image-unusable"],
            )
            self.assertEqual(outcome.diagnostics[0].path, anchor)

    def test_recursive_rebuild_keeps_model_isolation_and_skips_cache_symlinks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            root_face = root / "Root.face0.jpg"
            root_face.write_bytes(b"root")
            outside_cache = temporary_path / "outside.npy"
            outside_cache.write_bytes(b"outside")
            root_cache = root / "Root.face0.jpg.facenet512.npy"
            root_cache.symlink_to(outside_cache)
            other_model_cache = root / "Root.face0.jpg.arcface.npy"
            other_model_cache.write_bytes(b"ArcFace remains isolated")
            descendant = root / "Descendant"
            descendant.mkdir()
            descendant_face = descendant / "Nested.face1.jpg"
            descendant_face.write_bytes(b"nested")
            descendant_cache = descendant / "Nested.face1.jpg.facenet512.npy"
            np.save(descendant_cache, np.asarray(unit_vector(0)))

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root, recursive=True),
                DeterministicRecognition(
                    {
                        root_face: unit_vector(1),
                        descendant_face: unit_vector(2),
                    }
                ),
            )

            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.rebuilt, (descendant_cache,))
            self.assertTrue(root_cache.is_symlink())
            self.assertEqual(outside_cache.read_bytes(), b"outside")
            self.assertEqual(
                other_model_cache.read_bytes(),
                b"ArcFace remains isolated",
            )
            np.testing.assert_array_equal(np.load(descendant_cache), unit_vector(2))
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["maintenance-symlink-skipped"],
            )
            self.assertEqual(outcome.diagnostics[0].path, root_cache)

    def test_default_scope_rebuilds_only_static_images_for_the_selected_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            static_webp = root / "Still.face0.webp"
            Image.new("RGB", (2, 2), "red").save(static_webp, format="WEBP")
            animated_webp = root / "Motion.face1.webp"
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
            other_model_cache = root / "Still.face0.webp.facenet512.npy"
            other_model_cache.write_bytes(b"Facenet512 remains isolated")
            descendant = root / "Descendant"
            descendant.mkdir()
            descendant_face = descendant / "Nested.face2.jpg"
            descendant_face.write_bytes(b"nested")

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=root, model_name="ArcFace"),
                DeterministicRecognition({static_webp: unit_vector(1)}),
            )

            static_cache = root / "Still.face0.webp.arcface.npy"
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.rebuilt, (static_cache,))
            np.testing.assert_array_equal(np.load(static_cache), unit_vector(1))
            self.assertFalse((root / "Motion.face1.webp.arcface.npy").exists())
            self.assertFalse((descendant / "Nested.face2.jpg.arcface.npy").exists())
            self.assertEqual(
                other_model_cache.read_bytes(),
                b"Facenet512 remains isolated",
            )
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["animated-webp-unsupported"],
            )
            self.assertEqual(outcome.diagnostics[0].path, animated_webp)


if __name__ == "__main__":
    unittest.main()
