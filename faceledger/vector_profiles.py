"""Fixed recognition-vector compatibility profiles."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class VectorProfile:
    model_name: str
    cache_slug: str
    expected_dimensions: int
    cosine_threshold: float
    detector_backend: str = "retinaface"
    align: bool = True


DEFAULT_MODEL_NAME = "Facenet512"

VECTOR_PROFILES: Mapping[str, VectorProfile] = MappingProxyType(
    {
        "Facenet512": VectorProfile(
            model_name="Facenet512",
            cache_slug="facenet512",
            expected_dimensions=512,
            cosine_threshold=0.30,
        ),
        "ArcFace": VectorProfile(
            model_name="ArcFace",
            cache_slug="arcface",
            expected_dimensions=512,
            cosine_threshold=0.68,
        ),
    }
)
