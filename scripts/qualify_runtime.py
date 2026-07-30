"""Qualify Faceledger's locked DeepFace runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from faceledger.comparison import ComparisonRequest, Diagnostic, compare
from faceledger.maintenance import (
    CacheBuildRequest,
    build_vector_cache,
    rebuild_vector_cache,
)
from faceledger.trash import TrashRequest, trash_vector_cache
from faceledger.vector_profiles import DEFAULT_MODEL_NAME, VECTOR_PROFILES


class RuntimeQualificationContract(TypedDict):
    deepface_version: str
    python_version: str
    tensorflow_version: str
    tf_keras_version: str
    detector_backend: str
    align: bool
    cpu_only: bool
    recognition_models: list[str]
    static_image_formats: list[str]
    embedding_dimensions: dict[str, int]
    cache_slugs: dict[str, str]
    cosine_thresholds: dict[str, float]


_DEFAULT_PROFILE = VECTOR_PROFILES[DEFAULT_MODEL_NAME]
_REQUIRED_ASSETS = {
    "facenet512_weights.h5",
    "arcface_weights.h5",
    "retinaface.h5",
}

VECTOR_PROFILE: RuntimeQualificationContract = {
    "deepface_version": "0.0.100",
    "python_version": "3.12.13",
    "tensorflow_version": "2.21.0",
    "tf_keras_version": "2.21.0",
    "detector_backend": _DEFAULT_PROFILE.detector_backend,
    "align": _DEFAULT_PROFILE.align,
    "cpu_only": True,
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


def _runtime_contract_errors(
    runtime: dict[str, Any], wheel_tags: dict[str, list[str]]
) -> list[str]:
    """Identify installed runtime details outside the supported v1 envelope."""

    errors: list[str] = []
    expected_versions = {
        "python": VECTOR_PROFILE["python_version"],
        "tensorflow": VECTOR_PROFILE["tensorflow_version"],
        "tf_keras": VECTOR_PROFILE["tf_keras_version"],
    }
    for name, expected in expected_versions.items():
        actual = runtime[name]
        if actual != expected:
            errors.append(f"{name} is {actual}, expected {expected}")
    if runtime["machine"] != "x86_64":
        errors.append(f"machine is {runtime['machine']}, expected x86_64")
    libc_name, libc_version = runtime["libc"]
    if libc_name != "glibc":
        errors.append(f"libc is {libc_name or 'unknown'}, expected glibc")
    else:
        try:
            version = tuple(int(part) for part in libc_version.split("."))
        except ValueError:
            errors.append(f"glibc version is not numeric: {libc_version}")
        else:
            if version < (2, 27):
                errors.append(f"glibc is {libc_version}, expected 2.27 or newer")
    tensorflow_tags = wheel_tags.get("tensorflow", [])
    if "cp312-cp312-manylinux_2_27_x86_64" not in tensorflow_tags:
        errors.append(
            "TensorFlow is not installed from the qualified "
            "cp312-cp312-manylinux_2_27_x86_64 wheel"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        errors.append("CUDA visibility is not disabled")
    return errors


def _require(condition: bool, message: str) -> None:
    """Fail one release-qualification assertion with a readable reason."""

    if not condition:
        raise ValueError(message)


def _acquisition_notices(diagnostics: Sequence[Diagnostic]) -> list[str]:
    """Collect dependency assets announced by a public operation."""

    return [
        diagnostic.path.name
        for diagnostic in diagnostics
        if diagnostic.code == "model-asset-acquisition" and diagnostic.path is not None
    ]


def _qualify_public_operations(images: dict[str, Path]) -> list[dict[str, Any]]:
    """Exercise every public operation for each supported model and format."""

    checks: list[dict[str, Any]] = []
    suffixes = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    previous_xdg_data_home = os.environ.get("XDG_DATA_HOME")
    try:
        for model in VECTOR_PROFILE["recognition_models"]:
            for image_format in VECTOR_PROFILE["static_image_formats"]:
                check: dict[str, Any] = {
                    "model": model,
                    "format": image_format,
                    "operations": [],
                    "acquisition_notices": [],
                }
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="faceledger-release-qualification-"
                    ) as temporary_directory:
                        workspace = Path(temporary_directory)
                        target_root = workspace / "face-tree"
                        target_root.mkdir()
                        os.environ["XDG_DATA_HOME"] = str(workspace / "xdg-data")
                        target_image = target_root / (
                            f"candidate.face0{suffixes[image_format]}"
                        )
                        shutil.copyfile(images[image_format], target_image)

                        comparison_request = ComparisonRequest(
                            target_root=target_root,
                            source=images[image_format],
                            model_name=model,
                            threshold=2.0,
                            single_target_folder=True,
                            reuse_cache=False,
                        )
                        comparison = compare(comparison_request)
                        _require(
                            comparison.successful and comparison.complete,
                            "uncached comparison was not successful and complete",
                        )
                        _require(
                            len(comparison.matches) == 1,
                            f"expected one candidate match, got {len(comparison.matches)}",
                        )
                        check["acquisition_notices"].extend(
                            _acquisition_notices(comparison.diagnostics)
                        )
                        check["operations"].append("compare-uncached")

                        maintenance_request = CacheBuildRequest(
                            root=target_root,
                            model_name=model,
                        )
                        build = build_vector_cache(maintenance_request)
                        _require(
                            build.successful and build.complete,
                            "cache build was not successful and complete",
                        )
                        _require(
                            len(build.created) == 1,
                            f"expected one cache entry, got {len(build.created)}",
                        )
                        check["acquisition_notices"].extend(
                            _acquisition_notices(build.diagnostics)
                        )
                        check["operations"].append("cache-build")

                        cached_comparison = compare(
                            ComparisonRequest(
                                target_root=target_root,
                                source=images[image_format],
                                model_name=model,
                                threshold=2.0,
                                single_target_folder=True,
                            )
                        )
                        _require(
                            cached_comparison.successful and cached_comparison.complete,
                            "cached comparison was not successful and complete",
                        )
                        _require(
                            len(cached_comparison.matches) == 1,
                            "cached comparison did not return one candidate match",
                        )
                        _require(
                            len(cached_comparison.progress) == 1
                            and cached_comparison.progress[0].category == "source",
                            "cached comparison recalculated the target vector",
                        )
                        check["acquisition_notices"].extend(
                            _acquisition_notices(cached_comparison.diagnostics)
                        )
                        check["operations"].append("compare-cached")

                        rebuild = rebuild_vector_cache(maintenance_request)
                        _require(
                            rebuild.successful and rebuild.complete,
                            "cache rebuild was not successful and complete",
                        )
                        _require(
                            len(rebuild.rebuilt) == 1,
                            f"expected one rebuilt cache, got {len(rebuild.rebuilt)}",
                        )
                        check["acquisition_notices"].extend(
                            _acquisition_notices(rebuild.diagnostics)
                        )
                        check["operations"].append("cache-rebuild")

                        trash = trash_vector_cache(
                            TrashRequest(root=target_root, model_name=model)
                        )
                        _require(
                            trash.successful and trash.complete,
                            "trash was not successful and complete",
                        )
                        _require(
                            len(trash.moved) == 1 and trash.manifest_path is not None,
                            f"expected one recoverable cache move, got {len(trash.moved)}",
                        )
                        check["operations"].append("trash")
                    check["acquisition_notices"] = list(
                        dict.fromkeys(check["acquisition_notices"])
                    )
                    check["status"] = "passed"
                except Exception as error:  # noqa: BLE001 - report every attempted case
                    check.update(
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                checks.append(check)
    finally:
        if previous_xdg_data_home is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = previous_xdg_data_home
    return checks


def _qualify(images: dict[str, Path], report_path: Path, phase: str) -> int:
    """Run one release-qualification phase and persist its evidence report."""

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from deepface import DeepFace  # type: ignore[import-untyped]
    from PIL import Image

    assets_before = _asset_inventory()
    present_assets_before = {asset["path"] for asset in assets_before}
    if phase == "first-use" and _REQUIRED_ASSETS & present_assets_before:
        raise ValueError(
            "first-use qualification requires a fresh DEEPFACE_HOME without "
            "Facenet512, ArcFace, or RetinaFace assets"
        )
    if phase == "offline" and not _REQUIRED_ASSETS <= present_assets_before:
        missing = _REQUIRED_ASSETS - present_assets_before
        raise ValueError(
            "offline qualification requires acquired assets: "
            f"{', '.join(sorted(missing))}"
        )

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

    operation_checks = _qualify_public_operations(images)
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
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "libc": list(platform.libc_ver()),
        "tensorflow": importlib.metadata.version("tensorflow"),
        "tf_keras": importlib.metadata.version("tf-keras"),
        "installed_distributions": installed_distributions,
        "wheel_tags": wheel_tags,
    }
    runtime_contract_errors = _runtime_contract_errors(runtime, wheel_tags)
    assets = _asset_inventory()
    present_assets = {asset["path"] for asset in assets}
    acquisition_notices = {
        notice for check in operation_checks for notice in check["acquisition_notices"]
    }
    asset_lifecycle_errors: list[str] = []
    missing_assets = _REQUIRED_ASSETS - present_assets
    if missing_assets:
        asset_lifecycle_errors.append(
            f"required assets are missing: {', '.join(sorted(missing_assets))}"
        )
    if phase == "first-use" and acquisition_notices != _REQUIRED_ASSETS:
        missing_notices = _REQUIRED_ASSETS - acquisition_notices
        unexpected_notices = acquisition_notices - _REQUIRED_ASSETS
        if missing_notices:
            asset_lifecycle_errors.append(
                f"missing acquisition notices: {', '.join(sorted(missing_notices))}"
            )
        if unexpected_notices:
            asset_lifecycle_errors.append(
                "unexpected acquisition notices: "
                f"{', '.join(sorted(unexpected_notices))}"
            )
    if phase == "offline" and acquisition_notices:
        asset_lifecycle_errors.append(
            "offline operations announced missing assets: "
            f"{', '.join(sorted(acquisition_notices))}"
        )
    public_operations_passed = sum(
        check["status"] == "passed" for check in operation_checks
    )
    public_operations_failed = sum(
        check["status"] == "failed" for check in operation_checks
    )
    report = {
        **VECTOR_PROFILE,
        "phase": phase,
        "deepface_version": importlib.metadata.version("deepface"),
        "runtime": runtime,
        "runtime_contract_errors": runtime_contract_errors,
        "lock_sha256": _sha256(REPOSITORY_ROOT / "uv.lock"),
        "assets_before": assets_before,
        "assets": assets,
        "asset_lifecycle_errors": asset_lifecycle_errors,
        "checks": checks,
        "operation_checks": operation_checks,
        "summary": {
            "passed": passed,
            "failed": failed,
            "public_operations_passed": public_operations_passed,
            "public_operations_failed": public_operations_failed,
            "asset_lifecycle_failed": len(asset_lifecycle_errors),
            "runtime_contract_failed": len(runtime_contract_errors),
        },
    }
    report_path.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
    return (
        0
        if failed == 0
        and not public_operations_failed
        and not asset_lifecycle_errors
        and not runtime_contract_errors
        else 1
    )


def main() -> int:
    """Parse qualification arguments and run the requested action."""

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
    parser.add_argument(
        "--phase",
        choices=("first-use", "offline"),
        help="asset lifecycle phase exercised by --qualify",
    )
    arguments = parser.parse_args()

    if arguments.describe:
        print(json.dumps(VECTOR_PROFILE, indent=2))
        return 0

    if arguments.report is None:
        parser.error("--report is required with --qualify")
    if arguments.phase is None:
        parser.error("--phase is required with --qualify")
    images = _parse_images(arguments.image, parser)
    return _qualify(images, arguments.report, arguments.phase)


if __name__ == "__main__":
    raise SystemExit(main())
