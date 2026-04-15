"""Pydantic response schemas for the EcoMarineAI HTTP API.

Anything that touches the actual model lives in ``inference/``. This module
intentionally only contains FastAPI request/response shapes so it can be
imported without pulling in torch.
"""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


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
    rejection_reason: (
        Literal["not_a_marine_mammal", "low_confidence", "corrupted_image"] | None
    ) = None
    model_version: str = "effb4-arcface-v1"


# --- Webhook schemas ------------------------------------------------------

WebhookEvent = Literal["batch_completed", "prediction_rejected"]


class WebhookRegisterRequest(BaseModel):
    url: HttpUrl
    events: list[WebhookEvent] = Field(min_length=1)

    @field_validator("events")
    @classmethod
    def _unique_events(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("duplicate event names")
        return v


class WebhookRegisterResponse(BaseModel):
    webhook_id: str
    status: Literal["registered"] = "registered"
    url: HttpUrl
    events: list[WebhookEvent]


class WebhookInfo(BaseModel):
    webhook_id: str
    url: HttpUrl
    events: list[WebhookEvent]
    created_at: float


class WebhookListResponse(BaseModel):
    webhooks: list[WebhookInfo]
    count: int


# --- Export schemas -------------------------------------------------------


class ExportRecord(BaseModel):
    created_at: str
    image_ind: str
    class_animal: str
    id_animal: str
    probability: float
    is_cetacean: bool
    cetacean_score: float
    rejected: bool
    rejection_reason: str | None
    model_version: str
    bbox: list[int]


class ExportJSONResponse(BaseModel):
    count: int
    since: str | None = None
    records: list[ExportRecord]


def generate_base64_mask_with_removed_background(img_bytes: bytes) -> str | None:
    """Helper kept for backward compatibility. Returns None if rembg can't load
    (e.g. on newer Python where rembg calls sys.exit at import time).
    """
    import base64
    import logging

    try:
        from rembg import remove
    except (ImportError, SystemExit):
        logging.getLogger(__name__).warning("rembg unavailable; skipping mask.")
        return None
    try:
        return base64.b64encode(remove(img_bytes)).decode()
    except Exception:  # noqa: BLE001
        return None
