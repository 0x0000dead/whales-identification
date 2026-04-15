"""Pydantic response schemas for the EcoMarineAI HTTP API.

Anything that touches the actual model lives in ``inference/``. This module
intentionally only contains FastAPI request/response shapes so it can be
imported without pulling in torch.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single-image prediction result.

    Backward-compat note: the original schema only had the first six fields.
    The five new fields (``is_cetacean`` ... ``model_version``) are added with
    sensible defaults so legacy clients continue to deserialize.
    """

    image_ind: str
    bbox: list[int]
    class_animal: str
    id_animal: str
    probability: float
    mask: str | None = None

    is_cetacean: bool = True
    cetacean_score: float = Field(ge=0.0, le=1.0, default=1.0)
    rejected: bool = False
    rejection_reason: Literal[
        "not_a_marine_mammal", "low_confidence", "corrupted_image"
    ] | None = None
    model_version: str = "vit_l32-v1"


def generate_base64_mask_with_removed_background(img_bytes: bytes) -> str:
    """Helper kept for backward compatibility — used by /predict endpoints when
    the inference pipeline cannot be loaded (e.g. in unit tests with mocks).
    """
    import base64

    from rembg import remove

    return base64.b64encode(remove(img_bytes)).decode()
