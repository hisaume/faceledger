import tempfile
import unittest
from pathlib import Path

from faceledger.comparison import (
    CandidateMatch,
    ComparisonRequest,
    compare,
)
from faceledger.presentation import render_matches


class DeterministicRecognition:
    def __init__(self, vectors: dict[Path, tuple[float, ...]]) -> None:
        self._vectors = vectors

    def vector_for(self, image_path: Path) -> tuple[float, ...]:
        return self._vectors[image_path]


def snapshot_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class StandaloneComparisonTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
