import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from faceledger.comparison import (
    ComparisonRequest,
    Diagnostic,
    RecognitionFailure,
    compare,
)
from faceledger.maintenance import (
    CacheBuildRequest,
    build_vector_cache,
    rebuild_vector_cache,
)
from faceledger.trash import TrashRequest, trash_vector_cache


class FailingRecognition:
    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        raise RecognitionFailure(f"No usable face in {image_path.name}.")


class SourceOnlyRecognition:
    def __init__(self, source: Path) -> None:
        self._source = source

    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        if image_path == self._source:
            return (1.0, 0.0)
        raise RecognitionFailure(f"No usable face in {image_path.name}.")


class StaticRecognition:
    def vector_for(self, image_path: Path, profile: object) -> tuple[float, ...]:
        return (1.0, 0.0)


class DiagnosticStreamingTests(unittest.TestCase):
    def test_comparison_streams_an_early_validation_error_into_its_outcome(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            observed: list[Diagnostic] = []

            outcome = compare(
                ComparisonRequest(
                    source=root / "source.jpg",
                    target_root=root / "targets",
                    model_name="Unsupported",
                ),
                on_diagnostic=observed.append,
            )

        self.assertFalse(outcome.successful)
        self.assertEqual(
            [item.code for item in observed], ["recognition-model-unsupported"]
        )
        self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_comparison_streams_every_pre_recognition_validation_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "targets"
            target_root.mkdir()
            invalid_folder = root / "invalid-folder"
            invalid_folder.mkdir()
            cases = (
                (
                    ComparisonRequest(
                        source=source,
                        target_root=target_root,
                        threshold=float("inf"),
                    ),
                    "match-threshold-invalid",
                ),
                (
                    ComparisonRequest(
                        source=source,
                        source_folder=invalid_folder,
                        target_root=target_root,
                    ),
                    "source-selection-ambiguous",
                ),
                (
                    ComparisonRequest(target_root=target_root),
                    "source-selection-required",
                ),
                (
                    ComparisonRequest(source=source, target_root=root / "missing"),
                    "target-root-invalid",
                ),
                (
                    ComparisonRequest(
                        source=root / "missing.jpg",
                        target_root=target_root,
                    ),
                    "source-image-invalid",
                ),
                (
                    ComparisonRequest(
                        source_folder=invalid_folder,
                        target_root=target_root,
                    ),
                    "source-folder-invalid",
                ),
            )

            for request, expected_code in cases:
                with self.subTest(code=expected_code):
                    observed: list[Diagnostic] = []
                    outcome = compare(request, on_diagnostic=observed.append)

                    self.assertFalse(outcome.successful)
                    self.assertEqual(
                        [item.code for item in observed],
                        [expected_code],
                    )
                    self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_comparison_streams_a_recognition_error_into_its_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "targets"
            target_root.mkdir()
            observed: list[Diagnostic] = []

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                FailingRecognition(),
                on_diagnostic=observed.append,
            )

        self.assertFalse(outcome.successful)
        self.assertEqual([item.code for item in observed], ["source-image-unusable"])
        self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_comparison_streams_item_warnings_in_outcome_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "targets"
            target_root.mkdir()
            (target_root / "Alice.face0.jpg").write_bytes(b"alice")
            (target_root / "Bob.face1.jpg").write_bytes(b"bob")
            observed: list[Diagnostic] = []

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                SourceOnlyRecognition(source),
                on_diagnostic=observed.append,
            )

        self.assertTrue(outcome.successful)
        self.assertEqual(
            [item.code for item in observed],
            ["target-face-unusable", "target-face-unusable"],
        )
        self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_comparison_streams_cancellation_into_its_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "targets"
            target_root.mkdir()
            observed: list[Diagnostic] = []

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                SourceOnlyRecognition(source),
                on_diagnostic=observed.append,
                cancellation_requested=lambda: True,
            )

        self.assertFalse(outcome.complete)
        self.assertEqual([item.code for item in observed], ["comparison-cancelled"])
        self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_comparison_streams_diagnostics_from_cache_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            source.write_bytes(b"source")
            target_root = root / "targets"
            target_root.mkdir()
            target = target_root / "Alice.face0.jpg"
            target.write_bytes(b"alice")
            (target_root / "Alice.face0.jpg.facenet512.npy").write_bytes(b"invalid")
            observed: list[Diagnostic] = []

            outcome = compare(
                ComparisonRequest(source=source, target_root=target_root),
                StaticRecognition(),
                on_diagnostic=observed.append,
            )

        self.assertTrue(outcome.successful)
        self.assertEqual([item.code for item in observed], ["target-cache-invalid"])
        self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_cache_build_streams_an_early_validation_error_into_its_outcome(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_root = Path(temporary_directory) / "missing"
            observed: list[Diagnostic] = []

            outcome = build_vector_cache(
                CacheBuildRequest(root=missing_root),
                on_diagnostic=observed.append,
            )

        self.assertFalse(outcome.successful)
        self.assertEqual([item.code for item in observed], ["maintenance-root-invalid"])
        self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_cache_rebuild_streams_an_early_validation_error_into_its_outcome(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_root = Path(temporary_directory) / "missing"
            observed: list[Diagnostic] = []

            outcome = rebuild_vector_cache(
                CacheBuildRequest(root=missing_root),
                on_diagnostic=observed.append,
            )

        self.assertFalse(outcome.successful)
        self.assertEqual([item.code for item in observed], ["maintenance-root-invalid"])
        self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_cache_trash_streams_an_early_validation_error_into_its_outcome(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_root = Path(temporary_directory) / "missing"
            observed: list[Diagnostic] = []

            outcome = trash_vector_cache(
                TrashRequest(root=missing_root),
                on_diagnostic=observed.append,
            )

        self.assertFalse(outcome.successful)
        self.assertEqual([item.code for item in observed], ["maintenance-root-invalid"])
        self.assertEqual(outcome.diagnostics, tuple(observed))

    def test_public_operations_propagate_diagnostic_callback_failures(self) -> None:
        def callback_that_raises(
            failure: RuntimeError,
        ) -> Callable[[Diagnostic], None]:
            def fail(_diagnostic: Diagnostic) -> None:
                raise failure

            return fail

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            operations: tuple[
                tuple[str, Callable[[Callable[[Diagnostic], None]], object]], ...
            ] = (
                (
                    "comparison",
                    lambda callback: compare(
                        ComparisonRequest(
                            source=root / "source.jpg",
                            target_root=root / "targets",
                            model_name="Unsupported",
                        ),
                        on_diagnostic=callback,
                    ),
                ),
                (
                    "cache-build",
                    lambda callback: build_vector_cache(
                        CacheBuildRequest(root=root / "missing"),
                        on_diagnostic=callback,
                    ),
                ),
                (
                    "cache-rebuild",
                    lambda callback: rebuild_vector_cache(
                        CacheBuildRequest(root=root / "missing"),
                        on_diagnostic=callback,
                    ),
                ),
                (
                    "trash",
                    lambda callback: trash_vector_cache(
                        TrashRequest(root=root / "missing"),
                        on_diagnostic=callback,
                    ),
                ),
            )

            for operation_name, operation in operations:
                with self.subTest(operation=operation_name):
                    failure = RuntimeError(f"{operation_name} presentation failed")

                    with self.assertRaises(RuntimeError) as raised:
                        operation(callback_that_raises(failure))

                    self.assertIs(raised.exception, failure)


if __name__ == "__main__":
    unittest.main()
