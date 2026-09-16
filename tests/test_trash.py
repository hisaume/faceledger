import errno
import json
import os
import shutil
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from unittest.mock import patch

from faceledger.trash import TrashRequest, trash_vector_cache


class VectorCacheTrashTests(unittest.TestCase):
    def test_empty_model_specific_selection_is_a_successful_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            unrelated_npy = root / "notes.npy"
            unrelated_npy.write_bytes(b"unrelated")
            other_model = root / "Person.face0.jpg.arcface.npy"
            other_model.write_bytes(b"ArcFace")
            wrong_case = root / "Person.face0.jpg.FACENET512.npy"
            wrong_case.write_bytes(b"wrong case")
            xdg_data_home = temporary_path / "xdg-data"

            with patch.dict(
                os.environ,
                {"XDG_DATA_HOME": str(xdg_data_home)},
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root, model_name="Facenet512")
                )

            self.assertTrue(outcome.successful)
            self.assertIsNone(outcome.action_directory)
            self.assertIsNone(outcome.manifest_path)
            self.assertEqual(outcome.moved, ())
            self.assertEqual(
                outcome.message,
                "No matching Facenet512 cache entries found",
            )
            self.assertFalse(xdg_data_home.exists())
            self.assertEqual(unrelated_npy.read_bytes(), b"unrelated")
            self.assertEqual(other_model.read_bytes(), b"ArcFace")
            self.assertEqual(wrong_case.read_bytes(), b"wrong case")

    def test_creates_a_planned_xdg_action_and_moves_only_root_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            first_cache = root / "Alice.face0.jpg.facenet512.npy"
            first_cache.write_bytes(b"Alice vector")
            second_cache = root / "Bob.face1.jpg.facenet512.npy"
            second_cache.write_bytes(b"Bob vector")
            unrelated = root / "notes.npy"
            unrelated.write_bytes(b"unrelated")
            descendant = root / "Descendant"
            descendant.mkdir()
            descendant_cache = descendant / "Carol.face2.jpg.facenet512.npy"
            descendant_cache.write_bytes(b"Carol vector")
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 28, 12, 34, 56, 123456, tzinfo=UTC)

            with patch.dict(
                os.environ,
                {"XDG_DATA_HOME": str(xdg_data_home)},
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root, model_name="Facenet512"),
                    now=lambda: fixed_time,
                )

            action = xdg_data_home / "faceledger" / "trash" / "20260728T123456.123456Z"
            first_destination = action / "files" / first_cache.name
            second_destination = action / "files" / second_cache.name
            manifest = action / "manifest.txt"
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.action_directory, action)
            self.assertEqual(outcome.manifest_path, manifest)
            self.assertEqual(
                outcome.moved,
                (first_destination, second_destination),
            )
            self.assertFalse(first_cache.exists())
            self.assertFalse(second_cache.exists())
            self.assertEqual(first_destination.read_bytes(), b"Alice vector")
            self.assertEqual(second_destination.read_bytes(), b"Bob vector")
            self.assertTrue(descendant_cache.exists())
            self.assertEqual(unrelated.read_bytes(), b"unrelated")
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8")),
                [
                    {
                        "status": "moved",
                        "original": str(first_cache),
                        "trash_relative": f"files/{first_cache.name}",
                        "reason": None,
                    },
                    {
                        "status": "moved",
                        "original": str(second_cache),
                        "trash_relative": f"files/{second_cache.name}",
                        "reason": None,
                    },
                ],
            )

    def test_recursive_action_preserves_relative_paths_and_suffixes_collisions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            root_cache = root / "Root.face0.jpg.arcface.npy"
            root_cache.write_bytes(b"root ArcFace")
            branch = root / "Album"
            branch.mkdir()
            branch_cache = branch / "Person.face1.jpg.arcface.npy"
            branch_cache.write_bytes(b"branch ArcFace")
            other_model = branch / "Person.face1.jpg.facenet512.npy"
            other_model.write_bytes(b"other model")
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 28, 13, 0, 1, 2, tzinfo=UTC)
            action_id = "20260728T130001.000002Z"
            trash_root = xdg_data_home / "faceledger" / "trash"
            (trash_root / action_id).mkdir(parents=True)
            (trash_root / f"{action_id}-1").mkdir()

            with patch.dict(
                os.environ,
                {"XDG_DATA_HOME": str(xdg_data_home)},
            ):
                outcome = trash_vector_cache(
                    TrashRequest(
                        root=root,
                        model_name="ArcFace",
                        recursive=True,
                    ),
                    now=lambda: fixed_time,
                )

            action = trash_root / f"{action_id}-2"
            expected_files = {
                Path("manifest.txt"),
                Path("files/Root.face0.jpg.arcface.npy"),
                Path("files/Album/Person.face1.jpg.arcface.npy"),
            }
            self.assertEqual(outcome.action_directory, action)
            self.assertEqual(
                {
                    path.relative_to(action)
                    for path in action.rglob("*")
                    if path.is_file()
                },
                expected_files,
            )
            self.assertFalse(root_cache.exists())
            self.assertFalse(branch_cache.exists())
            self.assertEqual(other_model.read_bytes(), b"other model")
            manifest = json.loads((action / "manifest.txt").read_text(encoding="utf-8"))
            self.assertEqual(
                [entry["trash_relative"] for entry in manifest],
                [
                    "files/Root.face0.jpg.arcface.npy",
                    "files/Album/Person.face1.jpg.arcface.npy",
                ],
            )

    def test_failed_same_filesystem_move_is_recorded_and_later_entries_continue(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            failed_cache = root / "Alice.face0.jpg.facenet512.npy"
            failed_cache.write_bytes(b"Alice vector")
            moved_cache = root / "Bob.face1.jpg.facenet512.npy"
            moved_cache.write_bytes(b"Bob vector")
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 28, 14, 30, 0, tzinfo=UTC)
            real_rename = Path.rename

            def fail_alice(source: Path, destination: Path) -> Path:
                if source == failed_cache:
                    raise OSError(errno.EACCES, "move denied", str(source))
                return real_rename(source, destination)

            with (
                patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg_data_home)}),
                patch("faceledger.trash.Path.rename", new=fail_alice),
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root),
                    now=lambda: fixed_time,
                )

            action = outcome.action_directory
            self.assertIsNotNone(action)
            assert action is not None
            failed_destination = action / "files" / failed_cache.name
            moved_destination = action / "files" / moved_cache.name
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.moved, (moved_destination,))
            self.assertTrue(failed_cache.exists())
            self.assertFalse(failed_destination.exists())
            self.assertFalse(moved_cache.exists())
            self.assertEqual(moved_destination.read_bytes(), b"Bob vector")
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["trash-entry-move-failed"],
            )
            self.assertEqual(outcome.diagnostics[0].path, failed_cache)
            manifest_path = outcome.manifest_path
            self.assertIsNotNone(manifest_path)
            assert manifest_path is not None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest[0]["status"], "failed")
            self.assertIn("move denied", manifest[0]["reason"])
            self.assertEqual(manifest[1]["status"], "moved")
            self.assertIsNone(manifest[1]["reason"])

    def test_cross_filesystem_copy_is_verified_before_removing_the_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            cache = root / "Person.face0.jpg.facenet512.npy"
            cache.write_bytes(b"cache bytes")
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 28, 15, 0, 0, tzinfo=UTC)
            real_rename = Path.rename

            def cross_filesystem(source: Path, destination: Path) -> Path:
                if source == cache:
                    raise OSError(errno.EXDEV, "cross-device link", str(source))
                return real_rename(source, destination)

            with (
                patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg_data_home)}),
                patch("faceledger.trash.Path.rename", new=cross_filesystem),
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root),
                    now=lambda: fixed_time,
                )

            action = outcome.action_directory
            self.assertIsNotNone(action)
            assert action is not None
            destination = action / "files" / cache.name
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.moved, (destination,))
            self.assertFalse(cache.exists())
            self.assertEqual(destination.read_bytes(), b"cache bytes")
            self.assertEqual(outcome.diagnostics, ())
            manifest_path = outcome.manifest_path
            self.assertIsNotNone(manifest_path)
            assert manifest_path is not None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest[0]["status"], "moved")
            self.assertIsNone(manifest[0]["reason"])

    def test_destination_setup_failure_is_recorded_and_other_branches_continue(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            first_branch = root / "A"
            second_branch = root / "B"
            first_branch.mkdir(parents=True)
            second_branch.mkdir()
            failed_cache = first_branch / "Alice.face0.jpg.facenet512.npy"
            failed_cache.write_bytes(b"Alice vector")
            moved_cache = second_branch / "Bob.face1.jpg.facenet512.npy"
            moved_cache.write_bytes(b"Bob vector")
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 28, 15, 30, 0, tzinfo=UTC)
            action = xdg_data_home / "faceledger" / "trash" / "20260728T153000.000000Z"
            blocked_parent = action / "files" / "A"
            real_mkdir = Path.mkdir

            def deny_first_destination(
                path: Path,
                mode: int = 0o777,
                parents: bool = False,
                exist_ok: bool = False,
            ) -> None:
                if path == blocked_parent:
                    raise PermissionError(errno.EACCES, "directory denied", str(path))
                real_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

            with (
                patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg_data_home)}),
                patch("faceledger.trash.Path.mkdir", new=deny_first_destination),
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root, recursive=True),
                    now=lambda: fixed_time,
                )

            moved_destination = action / "files" / "B" / moved_cache.name
            self.assertTrue(outcome.successful)
            self.assertEqual(outcome.moved, (moved_destination,))
            self.assertTrue(failed_cache.exists())
            self.assertFalse(moved_cache.exists())
            self.assertEqual(moved_destination.read_bytes(), b"Bob vector")
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["trash-entry-move-failed"],
            )
            manifest_path = outcome.manifest_path
            self.assertIsNotNone(manifest_path)
            assert manifest_path is not None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest[0]["status"], "failed")
            self.assertIn("directory denied", manifest[0]["reason"])
            self.assertEqual(manifest[1]["status"], "moved")

    def test_failed_cross_filesystem_copy_keeps_source_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            failed_cache = root / "Alice.face0.jpg.facenet512.npy"
            failed_cache.write_bytes(b"Alice vector")
            moved_cache = root / "Bob.face1.jpg.facenet512.npy"
            moved_cache.write_bytes(b"Bob vector")
            selected = {failed_cache, moved_cache}
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 28, 16, 0, 0, tzinfo=UTC)
            real_rename = Path.rename
            real_copy = shutil.copyfileobj

            def cross_filesystem(source: Path, destination: Path) -> Path:
                if source in selected:
                    raise OSError(errno.EXDEV, "cross-device link", str(source))
                return real_rename(source, destination)

            def fail_alice_copy(
                source_file: BinaryIO,
                destination_file: BinaryIO,
                length: int = 0,
            ) -> None:
                if Path(source_file.name) == failed_cache:
                    raise OSError(errno.EIO, "copy failed", str(failed_cache))
                real_copy(source_file, destination_file, length)

            with (
                patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg_data_home)}),
                patch("faceledger.trash.Path.rename", new=cross_filesystem),
                patch(
                    "faceledger.trash.shutil.copyfileobj",
                    new=fail_alice_copy,
                ),
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root),
                    now=lambda: fixed_time,
                )

            action = outcome.action_directory
            self.assertIsNotNone(action)
            assert action is not None
            failed_destination = action / "files" / failed_cache.name
            moved_destination = action / "files" / moved_cache.name
            self.assertEqual(outcome.moved, (moved_destination,))
            self.assertTrue(failed_cache.exists())
            self.assertFalse(failed_destination.exists())
            self.assertFalse(moved_cache.exists())
            self.assertEqual(moved_destination.read_bytes(), b"Bob vector")
            self.assertEqual(tuple(action.rglob("*.tmp")), ())
            self.assertEqual(
                [diagnostic.code for diagnostic in outcome.diagnostics],
                ["trash-entry-move-failed"],
            )
            manifest_path = outcome.manifest_path
            self.assertIsNotNone(manifest_path)
            assert manifest_path is not None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest[0]["status"], "failed")
            self.assertIn("copy failed", manifest[0]["reason"])
            self.assertEqual(manifest[1]["status"], "moved")

    def test_failed_copy_verification_keeps_source_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            failed_cache = root / "Alice.face0.jpg.facenet512.npy"
            failed_cache.write_bytes(b"Alice vector")
            moved_cache = root / "Bob.face1.jpg.facenet512.npy"
            moved_cache.write_bytes(b"Bob vector")
            selected = {failed_cache, moved_cache}
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 28, 16, 30, 0, tzinfo=UTC)
            real_rename = Path.rename
            real_copy = shutil.copyfileobj

            def cross_filesystem(source: Path, destination: Path) -> Path:
                if source in selected:
                    raise OSError(errno.EXDEV, "cross-device link", str(source))
                return real_rename(source, destination)

            def corrupt_alice_copy(
                source_file: BinaryIO,
                destination_file: BinaryIO,
                length: int = 0,
            ) -> None:
                if Path(source_file.name) == failed_cache:
                    destination_file.write(b"corrupt")
                    return
                real_copy(source_file, destination_file, length)

            with (
                patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg_data_home)}),
                patch("faceledger.trash.Path.rename", new=cross_filesystem),
                patch(
                    "faceledger.trash.shutil.copyfileobj",
                    new=corrupt_alice_copy,
                ),
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root),
                    now=lambda: fixed_time,
                )

            action = outcome.action_directory
            self.assertIsNotNone(action)
            assert action is not None
            failed_destination = action / "files" / failed_cache.name
            moved_destination = action / "files" / moved_cache.name
            self.assertEqual(outcome.moved, (moved_destination,))
            self.assertEqual(failed_cache.read_bytes(), b"Alice vector")
            self.assertFalse(failed_destination.exists())
            self.assertFalse(moved_cache.exists())
            self.assertEqual(moved_destination.read_bytes(), b"Bob vector")
            manifest_path = outcome.manifest_path
            self.assertIsNotNone(manifest_path)
            assert manifest_path is not None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest[0]["status"], "failed")
            self.assertIn("verification failed", manifest[0]["reason"])
            self.assertEqual(manifest[1]["status"], "moved")

    def test_interrupted_entry_remains_planned_and_visibly_unconfirmed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            moved_cache = root / "Alice.face0.jpg.facenet512.npy"
            moved_cache.write_bytes(b"Alice vector")
            interrupted_cache = root / "Bob.face1.jpg.facenet512.npy"
            interrupted_cache.write_bytes(b"Bob vector")
            xdg_data_home = temporary_path / "xdg-data"
            fixed_time = datetime(2026, 7, 28, 17, 0, 0, tzinfo=UTC)
            action = xdg_data_home / "faceledger" / "trash" / "20260728T170000.000000Z"
            real_rename = Path.rename

            def interrupt_bob(source: Path, destination: Path) -> Path:
                if source == interrupted_cache:
                    raise KeyboardInterrupt
                return real_rename(source, destination)

            with (
                patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg_data_home)}),
                patch("faceledger.trash.Path.rename", new=interrupt_bob),
                self.assertRaises(KeyboardInterrupt),
            ):
                trash_vector_cache(
                    TrashRequest(root=root),
                    now=lambda: fixed_time,
                )

            manifest = json.loads((action / "manifest.txt").read_text(encoding="utf-8"))
            self.assertEqual(manifest[0]["status"], "moved")
            self.assertEqual(manifest[1]["status"], "planned")
            self.assertIsNone(manifest[1]["reason"])
            self.assertFalse(moved_cache.exists())
            self.assertTrue(interrupted_cache.exists())

    def test_uses_the_standard_local_share_fallback_without_xdg_data_home(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            home = temporary_path / "home"
            home.mkdir()
            root = temporary_path / "Face Tree"
            root.mkdir()
            cache = root / "Person.face0.jpg.facenet512.npy"
            cache.write_bytes(b"vector")
            fixed_time = datetime(2026, 7, 28, 14, 0, 0, 0, tzinfo=UTC)

            with patch.dict(
                os.environ,
                {"HOME": str(home)},
                clear=True,
            ):
                outcome = trash_vector_cache(
                    TrashRequest(root=root),
                    now=lambda: fixed_time,
                )

            self.assertEqual(
                outcome.action_directory,
                home
                / ".local"
                / "share"
                / "faceledger"
                / "trash"
                / "20260728T140000.000000Z",
            )


if __name__ == "__main__":
    unittest.main()
