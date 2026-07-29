"""Qualify Faceledger's locked DeepFace runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, TypedDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from faceledger.vector_profiles import DEFAULT_MODEL_NAME, VECTOR_PROFILES


class RuntimeQualificationContract(TypedDict):
    deepface_version: str
    python_version: str
    tensorflow_version: str
    tf_keras_version: str
    detector_backend: str
    align: bool
    recognition_models: list[str]
    static_image_formats: list[str]
    embedding_dimensions: dict[str, int]
    cache_slugs: dict[str, str]
    cosine_thresholds: dict[str, float]


_DEFAULT_PROFILE = VECTOR_PROFILES[DEFAULT_MODEL_NAME]

VECTOR_PROFILE: RuntimeQualificationContract = {
    "deepface_version": "0.0.100",
    "python_version": "3.12",
    "tensorflow_version": "2.21.0",
    "tf_keras_version": "2.21.0",
    "detector_backend": _DEFAULT_PROFILE.detector_backend,
    "align": _DEFAULT_PROFILE.align,
    "recognition_models": list(VECTOR_PROFILES),
    "static_image_formats": ["JPEG", "PNG", "WEBP"],
    "embedding_dimensions": {
        name: profile.expected_dimensions for name, profile in VECTOR_PROFILES.items()
    },
    "cache_slugs": {
        name: profile.cache_slug for name, profile in VECTOR_PROFILES.items()
    },
    "cosine_thresholds": {
        name: profile.cosine_threshold for name, profile in VECTOR_PROFILES.items()
    },
}


def _parse_images(
    values: list[str], parser: argparse.ArgumentParser
) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for value in values:
        image_format, separator, path = value.partition("=")
        if not separator or image_format not in VECTOR_PROFILE["static_image_formats"]:
            parser.error("--image must be one of JPEG=PATH, PNG=PATH, or WEBP=PATH")
        images[image_format] = Path(path)

    missing_formats = set(VECTOR_PROFILE["static_image_formats"]) - images.keys()
    if missing_formats:
        parser.error(f"missing image formats: {', '.join(sorted(missing_formats))}")
    return images


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_inventory() -> list[dict[str, Any]]:
    deepface_home = Path(os.environ.get("DEEPFACE_HOME", Path.home()))
    weights_path = deepface_home / ".deepface" / "weights"
    if not weights_path.exists():
        return []
    return [
        {
            "path": str(path.relative_to(weights_path)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(weights_path.rglob("*"))
        if path.is_file()
    ]


def _installed_runtime() -> tuple[dict[str, str], dict[str, list[str]]]:
    versions: dict[str, str] = {}
    wheel_tags: dict[str, list[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        versions[name] = distribution.version
        wheel_metadata = distribution.read_text("WHEEL") or ""
        tags = [
            line.removeprefix("Tag: ")
            for line in wheel_metadata.splitlines()
            if line.startswith("Tag: ")
        ]
        if tags:
            normalized_name = name.lower().replace("_", "-")
            wheel_tags[normalized_name] = tags
    return versions, wheel_tags


def _qualify(images: dict[str, Path], report_path: Path) -> int:
    from deepface import DeepFace  # type: ignore[import-untyped]
    from PIL import Image

    for expected_format, image_path in images.items():
        with Image.open(image_path) as image:
            if image.format != expected_format:
                raise ValueError(
                    f"{image_path} is {image.format}, expected {expected_format}"
                )
            if expected_format == "WEBP" and (
                bool(getattr(image, "is_animated", False))
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError(f"{image_path} must be a one-frame static WebP")

    checks: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    for model in VECTOR_PROFILE["recognition_models"]:
        for image_format in VECTOR_PROFILE["static_image_formats"]:
            check: dict[str, Any] = {"model": model, "format": image_format}
            try:
                representations = DeepFace.represent(
                    img_path=str(images[image_format]),
                    model_name=model,
                    detector_backend=VECTOR_PROFILE["detector_backend"],
                    enforce_detection=True,
                    align=VECTOR_PROFILE["align"],
                )
                if len(representations) != 1:
                    raise ValueError(
                        f"expected one representation, got {len(representations)}"
                    )
                embedding = representations[0]["embedding"]
                expected_dimensions = VECTOR_PROFILE["embedding_dimensions"][model]
                if len(embedding) != expected_dimensions:
                    raise ValueError(
                        f"expected {expected_dimensions} dimensions, got {len(embedding)}"
                    )
                if not all(math.isfinite(float(value)) for value in embedding):
                    raise ValueError("embedding contains a non-finite value")
                check.update(status="passed", embedding_dimensions=len(embedding))
                passed += 1
            except Exception as error:  # noqa: BLE001 - qualification must report every attempted case
                check.update(status="failed", error=f"{type(error).__name__}: {error}")
                failed += 1
            checks.append(check)

    installed_distributions, wheel_tags = _installed_runtime()
    report = {
        **VECTOR_PROFILE,
        "deepface_version": importlib.metadata.version("deepface"),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "libc": list(platform.libc_ver()),
            "tensorflow": importlib.metadata.version("tensorflow"),
            "tf_keras": importlib.metadata.version("tf-keras"),
            "installed_distributions": installed_distributions,
            "wheel_tags": wheel_tags,
        },
        "lock_sha256": _sha256(REPOSITORY_ROOT / "uv.lock"),
        "assets": _asset_inventory(),
        "checks": checks,
        "summary": {"passed": passed, "failed": failed},
    }
    report_path.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--describe",
        action="store_true",
        help="write the qualification contract as JSON",
    )
    actions.add_argument(
        "--qualify",
        action="store_true",
        help="calculate embeddings for every model and static image format",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="FORMAT=PATH",
        help="qualification image for JPEG, PNG, or WEBP (repeat three times)",
    )
    parser.add_argument(
        "--report", type=Path, help="write the JSON qualification report"
    )
    arguments = parser.parse_args()

    if arguments.describe:
        print(json.dumps(VECTOR_PROFILE, indent=2))
        return 0

    if arguments.report is None:
        parser.error("--report is required with --qualify")
    images = _parse_images(arguments.image, parser)
    return _qualify(images, arguments.report)


if __name__ == "__main__":
    raise SystemExit(main())
