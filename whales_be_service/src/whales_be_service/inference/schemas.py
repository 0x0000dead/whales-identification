"""Internal dataclasses used by the inference pipeline.

These are deliberately separated from the FastAPI Pydantic models in
``response_models.py`` so the inference layer has no web framework dependency
and can be unit-tested in isolation.
"""

from dataclasses import dataclass
from enum import Enum


class RejectionReason(str, Enum):
    NOT_A_MARINE_MAMMAL = "not_a_marine_mammal"
    LOW_CONFIDENCE = "low_confidence"
    CORRUPTED_IMAGE = "corrupted_image"


@dataclass(frozen=True)
class GateResult:
    """Output of the CLIP zero-shot anti-fraud gate."""

    positive_score: float
    negative_score: float
    is_cetacean: bool
    margin: float


@dataclass(frozen=True)
class PredictionResult:
    """Output of the identification model (no rejection logic — raw inference)."""

    class_id: str
    species: str
    probability: float
    bbox: list[int]
    embedding: list[float] | None = None
