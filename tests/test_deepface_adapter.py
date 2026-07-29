import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from faceledger.comparison import ComparisonRequest, compare

MODEL_ASSETS = {
    "Facenet512": "facenet512_weights.h5",
    "ArcFace": "arcface_weights.h5",
}


class DeepFaceModule(types.ModuleType):
    DeepFace: object


def _weights_path(home: Path) -> Path:
    return home / ".deepface" / "weights"


def _install_assets(home: Path, model_name: str) -> None:
    weights_path = _weights_path(home)
    weights_path.mkdir(parents=True, exist_ok=True)
    (weights_path / MODEL_ASSETS[model_name]).write_bytes(b"model weights")
    (weights_path / "retinaface.h5").write_bytes(b"detector weights")


def _deepface_module(represent: object) -> DeepFaceModule:
    module = DeepFaceModule("deepface")
    module.DeepFace = types.SimpleNamespace(represent=represent)
    return module


def _comparison_paths(root: Path) -> tuple[Path, Path]:
    source = root / "source.jpg"
    source.write_bytes(b"source image")
    target_root = root / "Target Person"
    target_root.mkdir()
    (target_root / "folder.jpg").write_bytes(b"target image")
    return source, target_root


class DeepFaceAdapterTests(unittest.TestCase):
    def test_public_comparison_uses_the_selected_fixed_cpu_profile(self) -> None:
        for model_name in MODEL_ASSETS:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source, target_root = _comparison_paths(root)
                    deepface_home = root / "deepface-home"
                    _install_assets(deepface_home, model_name)

                    def represent(
                        _expected_model_name: str = model_name,
                        **arguments: object,
                    ) -> list[dict[str, object]]:
                        self.assertEqual(arguments["model_name"], _expected_model_name)
                        self.assertEqual(arguments["detector_backend"], "retinaface")
                        self.assertIs(arguments["enforce_detection"], True)
                        self.assertIs(arguments["align"], True)
                        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
                        return [{"embedding": [1.0] + [0.0] * 511}]

                    with (
                        patch.dict(
                            os.environ,
                            {"DEEPFACE_HOME": str(deepface_home)},
                        ),
                        patch.dict(
                            sys.modules,
                            {"deepface": _deepface_module(represent)},
                        ),
                    ):
                        outcome = compare(
                            ComparisonRequest(
                                source=source,
                                target_root=target_root,
                                model_name=model_name,
                                reuse_cache=False,
                            )
                        )

                self.assertTrue(outcome.successful)
                self.assertEqual(len(outcome.matches), 1)
                self.assertEqual(outcome.matches[0].identity_path, Path("."))
                self.assertEqual(outcome.matches[0].cosine_distance, 0.0)

    def test_rejects_face_counts_and_dimensions_outside_the_profile(self) -> None:
        cases: tuple[tuple[list[dict[str, object]], str], ...] = (
            ([], "exactly one face"),
            (
                [
                    {"embedding": [1.0] + [0.0] * 511},
                    {"embedding": [1.0] + [0.0] * 511},
                ],
                "exactly one face",
            ),
            ([{"embedding": [1.0] + [0.0] * 510}], "512 dimensions"),
        )
        for representations, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "source.jpg"
                    source.write_bytes(b"source image")
                    target_root = root / "targets"
                    target_root.mkdir()
                    deepface_home = root / "deepface-home"
                    _install_assets(deepface_home, "Facenet512")

                    with (
                        patch.dict(
                            os.environ,
                            {"DEEPFACE_HOME": str(deepface_home)},
                        ),
                        patch.dict(
                            sys.modules,
                            {
                                "deepface": _deepface_module(
                                    lambda _representations=representations, **_arguments: (
                                        _representations
                                    )
                                )
                            },
                        ),
                    ):
                        outcome = compare(
                            ComparisonRequest(
                                source=source,
                                target_root=target_root,
                                reuse_cache=False,
                            )
                        )

                self.assertFalse(outcome.successful)
                self.assertEqual(
                    [diagnostic.code for diagnostic in outcome.diagnostics],
                    ["source-image-unusable"],
                )
                self.assertIn(expected_message, outcome.diagnostics[0].message)

    def test_announces_missing_dependency_assets_before_deepface_acquires_them(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, target_root = _comparison_paths(root)
            deepface_home = root / "deepface-home"

            def represent(**_arguments: object) -> list[dict[str, object]]:
                _install_assets(deepface_home, "Facenet512")
                return [{"embedding": [1.0] + [0.0] * 511}]

            with (
                patch.dict(
                    os.environ,
                    {"DEEPFACE_HOME": str(deepface_home)},
                ),
                patch.dict(
                    sys.modules,
                    {"deepface": _deepface_module(represent)},
                ),
            ):
                outcome = compare(
                    ComparisonRequest(
                        source=source,
                        target_root=target_root,
                        reuse_cache=False,
                    )
                )

        acquisition_notices = [
            diagnostic
            for diagnostic in outcome.diagnostics
            if diagnostic.code == "model-asset-acquisition"
        ]
        self.assertTrue(outcome.successful)
        acquisition_paths = [diagnostic.path for diagnostic in acquisition_notices]
        self.assertNotIn(None, acquisition_paths)
        self.assertEqual(
            {path.name for path in acquisition_paths if path is not None},
            {"facenet512_weights.h5", "retinaface.h5"},
        )
        self.assertTrue(
            all(diagnostic.severity == "info" for diagnostic in acquisition_notices)
        )

    def test_acquisition_failure_is_actionable_and_installed_assets_work_offline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, target_root = _comparison_paths(root)
            deepface_home = root / "deepface-home"

            with (
                patch.dict(
                    os.environ,
                    {"DEEPFACE_HOME": str(deepface_home)},
                ),
                patch.dict(
                    sys.modules,
                    {
                        "deepface": _deepface_module(
                            lambda **_arguments: (_ for _ in ()).throw(
                                ConnectionError("network unavailable")
                            )
                        )
                    },
                ),
            ):
                failed = compare(
                    ComparisonRequest(
                        source=source,
                        target_root=target_root,
                        reuse_cache=False,
                    )
                )

            _install_assets(deepface_home, "Facenet512")

            def offline_represent(**_arguments: object) -> list[dict[str, object]]:
                if not all(
                    path.is_file()
                    for path in (
                        _weights_path(deepface_home) / "facenet512_weights.h5",
                        _weights_path(deepface_home) / "retinaface.h5",
                    )
                ):
                    raise AssertionError("offline inference requires installed assets")
                return [{"embedding": [1.0] + [0.0] * 511}]

            with (
                patch.dict(
                    os.environ,
                    {"DEEPFACE_HOME": str(deepface_home)},
                ),
                patch.dict(
                    sys.modules,
                    {"deepface": _deepface_module(offline_represent)},
                ),
            ):
                offline = compare(
                    ComparisonRequest(
                        source=source,
                        target_root=target_root,
                        reuse_cache=False,
                    )
                )

        self.assertFalse(failed.successful)
        self.assertEqual(failed.diagnostics[-1].code, "model-assets-unavailable")
        self.assertIn("Retry", failed.diagnostics[-1].message)
        self.assertIn("offline", failed.diagnostics[-1].message)
        self.assertTrue(offline.successful)
        self.assertEqual(len(offline.matches), 1)
        self.assertNotIn(
            "model-asset-acquisition",
            [diagnostic.code for diagnostic in offline.diagnostics],
        )


if __name__ == "__main__":
    unittest.main()
