import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import FrameType

import numpy as np
from PIL import Image

from faceledger.cli import main
from faceledger.vector_profiles import VectorProfile


def process_environment(root: Path) -> tuple[dict[str, str], Path]:
    fake_dependency = root / "fake-dependency"
    fake_dependency.mkdir()
    (fake_dependency / "deepface.py").write_text(
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "class DeepFace:\n"
        "    @staticmethod\n"
        "    def represent(**_arguments):\n"
        "        Path(os.environ['FACELEDGER_TEST_MARKER']).write_text(\n"
        "            'working', encoding='utf-8'\n"
        "        )\n"
        "        time.sleep(0.2)\n"
        "        return [{'embedding': [1.0] + [0.0] * 511}]\n",
        encoding="utf-8",
    )
    deepface_home = root / "deepface-home"
    weights = deepface_home / ".deepface" / "weights"
    weights.mkdir(parents=True)
    (weights / "facenet512_weights.h5").write_bytes(b"model")
    (weights / "retinaface.h5").write_bytes(b"detector")
    marker = root / "recognition-started"
    environment = dict(os.environ)
    environment["DEEPFACE_HOME"] = str(deepface_home)
    environment["FACELEDGER_TEST_MARKER"] = str(marker)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(fake_dependency)
        if existing_python_path is None
        else f"{fake_dependency}{os.pathsep}{existing_python_path}"
    )
    return environment, marker


def run_interrupted_process(
    arguments: list[str],
    *,
    environment: dict[str, str],
    marker: Path,
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        [sys.executable, "-m", "faceledger", *arguments],
        cwd=Path(__file__).parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                raise AssertionError("recognition process did not start in time")
            time.sleep(0.001)
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"recognition process exited before SIGINT: {stdout!r} {stderr!r}"
            )
        os.kill(process.pid, signal.SIGINT)
        os.kill(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        returncode = process.returncode
        if returncode is None:
            raise AssertionError("recognition process has no return status")
        return returncode, stdout, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


class RepeatedSigintRecognition:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> tuple[float, ...]:
        self.calls.append(image_path)
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)
        return (1.0,) + (0.0,) * (profile.expected_dimensions - 1)


class FailOnceStringIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    def write(self, text: str) -> int:
        if not self._failed:
            self._failed = True
            raise OSError("simulated console failure")
        return super().write(text)


class CliCancellationTests(unittest.TestCase):
    def test_repeated_sigint_cancels_comparison_and_restores_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            (target_root / "Person.face0.jpg").write_bytes(b"target")
            result_path = root / "comparison.txt"
            log_path = root / "comparison.log"
            stdout = io.StringIO()
            stderr = io.StringIO()
            recognition = RepeatedSigintRecognition()
            previous_calls: list[int] = []

            def previous_handler(
                signal_number: int,
                frame: FrameType | None,
            ) -> None:
                previous_calls.append(signal_number)

            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, previous_handler)
            try:
                status = main(
                    [
                        "compare",
                        str(source),
                        str(target_root),
                        "--result-file",
                        str(result_path),
                        "--log-file",
                        str(log_path),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    recognition=recognition,
                )
                restored_handler = signal.getsignal(signal.SIGINT)
            finally:
                signal.signal(signal.SIGINT, original_handler)

            self.assertEqual(status, 130)
            self.assertEqual(previous_calls, [])
            self.assertIs(restored_handler, previous_handler)
            self.assertEqual(recognition.calls, [source])
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue().count("comparison-cancelled"), 1)
            self.assertFalse(result_path.exists())
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("Status: cancelled\n", log)
            self.assertEqual(log.count("comparison-cancelled"), 1)

    def test_sigint_cancels_cache_build_after_safely_completed_item(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            first_face = root / "Alice.face0.jpg"
            first_face.write_bytes(b"first")
            second_face = root / "Bob.face1.jpg"
            second_face.write_bytes(b"second")
            stdout = io.StringIO()
            stderr = io.StringIO()
            recognition = RepeatedSigintRecognition()
            previous_calls: list[int] = []

            def previous_handler(
                signal_number: int,
                frame: FrameType | None,
            ) -> None:
                previous_calls.append(signal_number)

            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, previous_handler)
            try:
                status = main(
                    ["cache", "build", str(root)],
                    stdout=stdout,
                    stderr=stderr,
                    recognition=recognition,
                )
            finally:
                signal.signal(signal.SIGINT, original_handler)

            first_cache = root / "Alice.face0.jpg.facenet512.npy"
            second_cache = root / "Bob.face1.jpg.facenet512.npy"
            self.assertEqual(status, 130)
            self.assertEqual(previous_calls, [])
            self.assertEqual(recognition.calls, [first_face])
            self.assertTrue(first_cache.is_file())
            self.assertFalse(second_cache.exists())
            self.assertIn("Status: cancelled\n", stdout.getvalue())
            self.assertIn("Created: 1\n", stdout.getvalue())
            self.assertIn("Retained: 0\n", stdout.getvalue())
            self.assertEqual(stderr.getvalue().count("cache-build-cancelled"), 1)

    def test_sigint_cancels_rebuild_without_replacing_unprocessed_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Face Tree"
            root.mkdir()
            first_face = root / "Alice.face0.jpg"
            first_face.write_bytes(b"first")
            first_cache = root / "Alice.face0.jpg.facenet512.npy"
            np.save(first_cache, np.asarray((0.0, 1.0) + (0.0,) * 510))
            second_face = root / "Bob.face1.jpg"
            second_face.write_bytes(b"second")
            second_cache = root / "Bob.face1.jpg.facenet512.npy"
            np.save(second_cache, np.asarray((0.0, 0.0, 1.0) + (0.0,) * 509))
            second_before = second_cache.read_bytes()
            stdout = io.StringIO()
            stderr = io.StringIO()
            recognition = RepeatedSigintRecognition()
            previous_calls: list[int] = []

            def previous_handler(
                signal_number: int,
                frame: FrameType | None,
            ) -> None:
                previous_calls.append(signal_number)

            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, previous_handler)
            try:
                status = main(
                    ["cache", "rebuild", str(root)],
                    stdout=stdout,
                    stderr=stderr,
                    recognition=recognition,
                )
            finally:
                signal.signal(signal.SIGINT, original_handler)

            self.assertEqual(status, 130)
            self.assertEqual(previous_calls, [])
            self.assertEqual(recognition.calls, [first_face])
            np.testing.assert_array_equal(
                np.load(first_cache),
                (1.0,) + (0.0,) * 511,
            )
            self.assertEqual(second_cache.read_bytes(), second_before)
            self.assertIn("Status: cancelled\n", stdout.getvalue())
            self.assertIn("Rebuilt: 1\n", stdout.getvalue())
            self.assertEqual(stderr.getvalue().count("cache-rebuild-cancelled"), 1)

    def test_process_sigint_preserves_trash_manifest_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            root = temporary_path / "Face Tree"
            root.mkdir()
            caches = tuple(
                root / f"Person{index:03d}.face0.jpg.facenet512.npy"
                for index in range(200)
            )
            for cache in caches:
                cache.write_bytes(f"cache {cache.name}".encode())
            xdg_data_home = temporary_path / "xdg-data"
            environment = os.environ.copy()
            environment["XDG_DATA_HOME"] = str(xdg_data_home)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "faceledger",
                    "cache",
                    "trash",
                    str(root),
                    "--no-progress",
                ],
                cwd=Path(__file__).parents[1],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while caches[0].exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("trash process did not move its first item in time")
                    time.sleep(0.001)
                self.assertIsNone(process.poll())
                os.kill(process.pid, signal.SIGINT)
                os.kill(process.pid, signal.SIGINT)
                stdout, stderr = process.communicate(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

            self.assertEqual(process.returncode, 130)
            self.assertIn("Status: cancelled\n", stdout)
            self.assertEqual(stderr.count("trash-cancelled"), 1)
            self.assertIn("Recovery directory:", stderr)
            self.assertIn("Recovery manifest:", stderr)
            trash_root = xdg_data_home / "faceledger" / "trash"
            (action,) = tuple(trash_root.iterdir())
            manifest = json.loads((action / "manifest.txt").read_text(encoding="utf-8"))
            states = [entry["status"] for entry in manifest]
            self.assertIn("moved", states)
            self.assertIn("planned", states)
            self.assertNotIn("failed", states)

    def test_process_sigint_cancels_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            (target_root / "Person.face0.jpg").write_bytes(b"target")
            environment, marker = process_environment(root)

            status, stdout, stderr = run_interrupted_process(
                ["compare", str(source), str(target_root)],
                environment=environment,
                marker=marker,
            )

            self.assertEqual(status, 130)
            self.assertEqual(stdout, "")
            self.assertEqual(stderr.count("comparison-cancelled"), 1)
            self.assertNotIn("Traceback", stderr)

    def test_process_sigint_cancels_cache_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_face = root / "Alice.face0.jpg"
            first_face.write_bytes(b"first")
            second_face = root / "Bob.face1.jpg"
            second_face.write_bytes(b"second")
            environment, marker = process_environment(root)

            status, stdout, stderr = run_interrupted_process(
                ["cache", "build", str(root)],
                environment=environment,
                marker=marker,
            )

            self.assertEqual(status, 130)
            self.assertTrue((root / "Alice.face0.jpg.facenet512.npy").is_file())
            self.assertFalse((root / "Bob.face1.jpg.facenet512.npy").exists())
            self.assertIn("Status: cancelled\n", stdout)
            self.assertIn("Created: 1\n", stdout)
            self.assertEqual(stderr.count("cache-build-cancelled"), 1)

    def test_process_sigint_cancels_cache_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_face = root / "Alice.face0.jpg"
            first_face.write_bytes(b"first")
            first_cache = root / "Alice.face0.jpg.facenet512.npy"
            np.save(first_cache, np.asarray((0.0, 1.0) + (0.0,) * 510))
            second_face = root / "Bob.face1.jpg"
            second_face.write_bytes(b"second")
            second_cache = root / "Bob.face1.jpg.facenet512.npy"
            np.save(second_cache, np.asarray((0.0, 0.0, 1.0) + (0.0,) * 509))
            second_before = second_cache.read_bytes()
            environment, marker = process_environment(root)

            status, stdout, stderr = run_interrupted_process(
                ["cache", "rebuild", str(root)],
                environment=environment,
                marker=marker,
            )

            self.assertEqual(status, 130)
            np.testing.assert_array_equal(
                np.load(first_cache),
                (1.0,) + (0.0,) * 511,
            )
            self.assertEqual(second_cache.read_bytes(), second_before)
            self.assertIn("Status: cancelled\n", stdout)
            self.assertIn("Rebuilt: 1\n", stdout)
            self.assertEqual(stderr.count("cache-rebuild-cancelled"), 1)

    def test_cancelled_build_does_not_hide_console_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_face = root / "Alice.face0.jpg"
            first_face.write_bytes(b"first")
            (root / "Bob.face1.jpg").write_bytes(b"second")
            stdout = FailOnceStringIO()
            stderr = io.StringIO()
            recognition = RepeatedSigintRecognition()
            previous_calls: list[int] = []

            def previous_handler(
                signal_number: int,
                frame: FrameType | None,
            ) -> None:
                previous_calls.append(signal_number)

            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, previous_handler)
            try:
                status = main(
                    ["cache", "build", str(root)],
                    stdout=stdout,
                    stderr=stderr,
                    recognition=recognition,
                )
            finally:
                signal.signal(signal.SIGINT, original_handler)

            self.assertEqual(status, 1)
            self.assertEqual(previous_calls, [])
            self.assertEqual(recognition.calls, [first_face])
            self.assertIn("cache-build-cancelled", stderr.getvalue())
            self.assertIn("presentation-failure", stderr.getvalue())
            self.assertNotIn("internal-error", stderr.getvalue())

    def test_cancelled_build_does_not_hide_diagnostic_callback_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_face = root / "Alice.face0.jpg"
            first_face.write_bytes(b"first")
            (root / "Bob.face1.jpg").write_bytes(b"second")
            stdout = io.StringIO()
            stderr = FailOnceStringIO()
            recognition = RepeatedSigintRecognition()
            previous_calls: list[int] = []

            def previous_handler(
                signal_number: int,
                frame: FrameType | None,
            ) -> None:
                previous_calls.append(signal_number)

            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, previous_handler)
            try:
                status = main(
                    ["cache", "build", str(root)],
                    stdout=stdout,
                    stderr=stderr,
                    recognition=recognition,
                )
            finally:
                signal.signal(signal.SIGINT, original_handler)

            self.assertEqual(status, 1)
            self.assertEqual(previous_calls, [])
            self.assertEqual(recognition.calls, [first_face])
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("presentation-failure", stderr.getvalue())
            self.assertIn("simulated console failure", stderr.getvalue())
            self.assertNotIn("internal-error", stderr.getvalue())

    def test_cancelled_comparison_does_not_hide_log_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            Image.new("RGB", (2, 2), "white").save(source)
            target_root = root / "face tree"
            target_root.mkdir()
            (target_root / "Person.face0.jpg").write_bytes(b"target")
            result_path = root / "comparison.txt"
            log_path = root / "missing" / "comparison.log"
            stdout = io.StringIO()
            stderr = io.StringIO()
            recognition = RepeatedSigintRecognition()
            previous_calls: list[int] = []

            def previous_handler(
                signal_number: int,
                frame: FrameType | None,
            ) -> None:
                previous_calls.append(signal_number)

            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, previous_handler)
            try:
                status = main(
                    [
                        "compare",
                        str(source),
                        str(target_root),
                        "--result-file",
                        str(result_path),
                        "--log-file",
                        str(log_path),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    recognition=recognition,
                )
            finally:
                signal.signal(signal.SIGINT, original_handler)

            self.assertEqual(status, 1)
            self.assertEqual(previous_calls, [])
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(result_path.exists())
            self.assertFalse(log_path.exists())
            self.assertEqual(stderr.getvalue().count("comparison-cancelled"), 1)
            self.assertEqual(
                stderr.getvalue().count("log-artifact-write-failed"),
                1,
            )


if __name__ == "__main__":
    unittest.main()
