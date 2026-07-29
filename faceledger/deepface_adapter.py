"""Locked DeepFace recognition boundary."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from faceledger.comparison import AssetAcquisitionFailure, RecognitionFailure
from faceledger.vector_profiles import VectorProfile

_MODEL_ASSET_NAMES = {
    "Facenet512": "facenet512_weights.h5",
    "ArcFace": "arcface_weights.h5",
}
_DETECTOR_ASSET_NAME = "retinaface.h5"


class DeepFaceRecognition:
    """Calculate one embedding with Faceledger's locked DeepFace profile."""

    def __init__(self, announce_missing_asset: Callable[[Path], None]) -> None:
        self._announce_missing_asset = announce_missing_asset
        self._announced_assets: set[Path] = set()

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> Sequence[float]:
        """Calculate and validate an embedding with the locked CPU profile."""

        required_assets = self._required_assets(profile)
        missing_assets = tuple(path for path in required_assets if not path.is_file())
        for asset_path in missing_assets:
            if asset_path not in self._announced_assets:
                self._announce_missing_asset(asset_path)
                self._announced_assets.add(asset_path)

        # DeepFace imports TensorFlow. Fix its device visibility before that import
        # so this version-one runtime remains CPU-only.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            from deepface import DeepFace  # type: ignore[import-untyped]

            representations: Any = DeepFace.represent(
                img_path=str(image_path),
                model_name=profile.model_name,
                detector_backend=profile.detector_backend,
                enforce_detection=True,
                align=profile.align,
            )
        except Exception as error:
            if missing_assets and self._looks_like_acquisition_failure(
                error,
                required_assets,
            ):
                missing_names = ", ".join(path.name for path in missing_assets)
                raise AssetAcquisitionFailure(
                    "DeepFace could not acquire required model assets "
                    f"({missing_names}). Retry with network access available. "
                    "After acquisition succeeds, the installed assets can be used "
                    f"offline. Dependency error: {error}"
                ) from error
            raise RecognitionFailure(
                f"DeepFace could not produce a face embedding for {image_path.name}: "
                f"{error}"
            ) from error

        if not isinstance(representations, list) or len(representations) != 1:
            count = len(representations) if isinstance(representations, list) else 0
            raise RecognitionFailure(
                f"Expected exactly one face in {image_path.name}; detected {count}."
            )

        representation = representations[0]
        if not isinstance(representation, dict):
            raise RecognitionFailure(
                f"DeepFace returned an invalid representation for {image_path.name}."
            )
        embedding = representation.get("embedding")
        if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes)):
            raise RecognitionFailure(
                f"DeepFace returned no numeric embedding for {image_path.name}."
            )
        try:
            vector = tuple(float(value) for value in embedding)
        except (TypeError, ValueError, OverflowError) as error:
            raise RecognitionFailure(
                f"DeepFace returned a non-numeric embedding for {image_path.name}."
            ) from error
        if len(vector) != profile.expected_dimensions:
            raise RecognitionFailure(
                f"Expected {profile.expected_dimensions} dimensions for "
                f"{profile.model_name}; received {len(vector)} for {image_path.name}."
            )
        if not all(math.isfinite(value) for value in vector):
            raise RecognitionFailure(
                f"DeepFace returned a non-finite embedding for {image_path.name}."
            )
        return vector

    @staticmethod
    def _required_assets(profile: VectorProfile) -> tuple[Path, Path]:
        """Resolve the model and detector assets required by the profile."""

        deepface_home = Path(os.environ.get("DEEPFACE_HOME", Path.home()))
        weights_path = deepface_home / ".deepface" / "weights"
        return (
            weights_path / _MODEL_ASSET_NAMES[profile.model_name],
            weights_path / _DETECTOR_ASSET_NAME,
        )

    @staticmethod
    def _looks_like_acquisition_failure(
        error: Exception,
        required_assets: tuple[Path, Path],
    ) -> bool:
        """Distinguish asset acquisition failures from recognition failures."""

        if any(not path.is_file() for path in required_assets):
            return True
        messages: list[str] = []
        current: BaseException | None = error
        while current is not None:
            messages.append(str(current).lower())
            current = current.__cause__
        combined_message = " ".join(messages)
        return any(
            phrase in combined_message
            for phrase in (
                "download",
                "pre-trained weight",
                "pretrained weight",
            )
        )
