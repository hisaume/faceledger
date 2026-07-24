import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_COMMAND = REPOSITORY_ROOT / "scripts" / "qualify_runtime.py"


class RuntimeQualificationContractTests(unittest.TestCase):
    def test_describes_the_locked_vector_profile(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(QUALIFICATION_COMMAND), "--describe"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        contract = json.loads(completed.stdout)

        self.assertEqual(contract["deepface_version"], "0.0.100")
        self.assertEqual(contract["python_version"], "3.12")
        self.assertEqual(contract["tensorflow_version"], "2.21.0")
        self.assertEqual(contract["tf_keras_version"], "2.21.0")
        self.assertEqual(contract["detector_backend"], "retinaface")
        self.assertIs(contract["align"], True)
        self.assertEqual(
            contract["recognition_models"],
            ["Facenet512", "ArcFace"],
        )
        self.assertEqual(contract["static_image_formats"], ["JPEG", "PNG", "WEBP"])
        self.assertEqual(
            contract["embedding_dimensions"],
            {"Facenet512": 512, "ArcFace": 512},
        )

    def test_qualifies_every_model_and_static_image_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            adapter_path = temporary_path / "adapter"
            package_path = adapter_path / "deepface"
            package_path.mkdir(parents=True)
            (package_path / "__init__.py").write_text(
                """
from pathlib import Path


DIMENSIONS = {
    "Facenet512": 512,
    "ArcFace": 512,
}


class DeepFace:
    @staticmethod
    def represent(*, img_path, model_name, detector_backend, enforce_detection, align):
        if detector_backend != "retinaface":
            raise AssertionError("unexpected detector backend")
        if align is not True:
            raise AssertionError("alignment must be enabled")
        if enforce_detection is not True:
            raise AssertionError("face detection must be enforced")
        suffix = Path(img_path).suffix.lower()
        if suffix not in {".jpg", ".png", ".webp"}:
            raise AssertionError("unexpected image format")
        return [{"embedding": [0.5] * DIMENSIONS[model_name]}]
""".lstrip(),
                encoding="utf-8",
            )
            pillow_path = adapter_path / "PIL"
            pillow_path.mkdir()
            (pillow_path / "__init__.py").write_text(
                "from . import Image\n", encoding="utf-8"
            )
            (pillow_path / "Image.py").write_text(
                """
from pathlib import Path


class OpenedImage:
    def __init__(self, path):
        self.format = {".jpg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[Path(path).suffix]
        self.n_frames = 1
        self.is_animated = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def open(path):
    return OpenedImage(path)
""".lstrip(),
                encoding="utf-8",
            )
            distribution_path = adapter_path / "deepface-0.0.100.dist-info"
            distribution_path.mkdir()
            (distribution_path / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: deepface\nVersion: 0.0.100\n",
                encoding="utf-8",
            )
            (distribution_path / "WHEEL").write_text(
                "Wheel-Version: 1.0\nTag: py3-none-any\n",
                encoding="utf-8",
            )
            tensorflow_distribution_path = adapter_path / "tensorflow-2.21.0.dist-info"
            tensorflow_distribution_path.mkdir()
            (tensorflow_distribution_path / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: tensorflow\nVersion: 2.21.0\n",
                encoding="utf-8",
            )
            (tensorflow_distribution_path / "WHEEL").write_text(
                "Wheel-Version: 1.0\n"
                "Tag: cp312-cp312-manylinux_2_27_x86_64\n",
                encoding="utf-8",
            )
            tf_keras_distribution_path = adapter_path / "tf_keras-2.21.0.dist-info"
            tf_keras_distribution_path.mkdir()
            (tf_keras_distribution_path / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: tf-keras\nVersion: 2.21.0\n",
                encoding="utf-8",
            )
            (tf_keras_distribution_path / "WHEEL").write_text(
                "Wheel-Version: 1.0\nTag: py3-none-any\n",
                encoding="utf-8",
            )

            images = {}
            for image_format, suffix in (("JPEG", ".jpg"), ("PNG", ".png"), ("WEBP", ".webp")):
                image_path = temporary_path / f"face{suffix}"
                image_path.write_bytes(b"qualification fixture")
                images[image_format] = image_path

            report_path = temporary_path / "report.json"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(adapter_path)
            environment["DEEPFACE_HOME"] = str(temporary_path / "asset-home")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(QUALIFICATION_COMMAND),
                    "--qualify",
                    *(
                        argument
                        for image_format, image_path in images.items()
                        for argument in ("--image", f"{image_format}={image_path}")
                    ),
                    "--report",
                    str(report_path),
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(completed.stdout, "")
        self.assertEqual(report["deepface_version"], "0.0.100")
        self.assertEqual(report["assets"], [])
        self.assertEqual(report["runtime"]["machine"], "x86_64")
        self.assertIn("python", report["runtime"])
        self.assertIn("libc", report["runtime"])
        self.assertEqual(report["runtime"]["tensorflow"], "2.21.0")
        self.assertEqual(report["runtime"]["tf_keras"], "2.21.0")
        self.assertEqual(report["runtime"]["wheel_tags"]["deepface"], ["py3-none-any"])
        self.assertEqual(
            report["runtime"]["wheel_tags"]["tensorflow"],
            ["cp312-cp312-manylinux_2_27_x86_64"],
        )
        self.assertEqual(len(report["lock_sha256"]), 64)
        self.assertEqual(report["summary"], {"passed": 6, "failed": 0})
        self.assertEqual(len(report["checks"]), 6)
        self.assertEqual(
            {(check["model"], check["format"]) for check in report["checks"]},
            {
                (model, image_format)
                for model in ["Facenet512", "ArcFace"]
                for image_format in ["JPEG", "PNG", "WEBP"]
            },
        )
        self.assertTrue(all(check["status"] == "passed" for check in report["checks"]))
        self.assertTrue(
            all(
                check["embedding_dimensions"]
                == report["embedding_dimensions"][check["model"]]
                for check in report["checks"]
            )
        )


if __name__ == "__main__":
    unittest.main()
