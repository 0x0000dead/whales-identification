"""Inference pipeline orchestrator: anti-fraud gate → identification → response.

Public API: ``InferencePipeline.predict(pil_img, filename) -> Detection``.

This is the ONLY entry point used by ``main.py``. The previous design wired
random mock data directly in the route handler; that path has been removed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .anti_fraud import AntiFraudGate
from .identification import IdentificationModel
from .schemas import RejectionReason

if TYPE_CHECKING:
    from PIL import Image

    from ..response_models import Detection

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Orchestrates the two-stage cetacean recognition flow."""

    def __init__(
        self,
        anti_fraud: AntiFraudGate,
        identification: IdentificationModel,
        min_confidence: float = 0.05,
    ) -> None:
        self.anti_fraud = anti_fraud
        self.identification = identification
        self.min_confidence = min_confidence
        self.model_version = identification.model_version

    def warmup(self) -> None:
        """Force lazy-load of both stages. Called from FastAPI lifespan startup."""
        try:
            self.anti_fraud._load()  # noqa: SLF001
            logger.info("Anti-fraud gate warmed up.")
        except Exception as e:  # noqa: BLE001
            logger.warning("Anti-fraud gate warmup failed: %s", e)
        try:
            self.identification._load()  # noqa: SLF001
            logger.info("Identification model warmed up.")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Identification model warmup failed (running without ID stage): %s",
                e,
            )

    def predict(
        self,
        pil_img: "Image.Image",
        filename: str,
        img_bytes: bytes | None = None,
        generate_mask: bool = True,
    ) -> "Detection":
        """Run the full pipeline. Returns a ``Detection`` with all fields populated.

        ``img_bytes`` is optional and only used for the rembg background mask
        (which works on bytes, not PIL). If omitted, the mask is skipped.
        """
        from ..response_models import Detection  # noqa: PLC0415

        # Stage 1: CLIP zero-shot anti-fraud gate
        gate = self.anti_fraud.score(pil_img)
        if not gate.is_cetacean:
            return Detection(
                image_ind=filename,
                bbox=[0, 0, pil_img.width, pil_img.height],
                class_animal="",
                id_animal="unknown",
                probability=0.0,
                mask=None,
                is_cetacean=False,
                cetacean_score=round(gate.positive_score, 4),
                rejected=True,
                rejection_reason=RejectionReason.NOT_A_MARINE_MAMMAL.value,
                model_version=self.model_version,
            )

        # Stage 2: identification
        try:
            ident = self.identification.predict(pil_img)
        except FileNotFoundError as e:
            logger.warning(
                "Identification model unavailable; returning gate-only result: %s", e
            )
            return Detection(
                image_ind=filename,
                bbox=[0, 0, pil_img.width, pil_img.height],
                class_animal="",
                id_animal="cetacean_unidentified",
                probability=round(gate.positive_score, 4),
                mask=None,
                is_cetacean=True,
                cetacean_score=round(gate.positive_score, 4),
                rejected=False,
                rejection_reason=None,
                model_version=self.model_version,
            )

        # Stage 3: confidence threshold gating
        if ident.probability < self.min_confidence:
            return Detection(
                image_ind=filename,
                bbox=ident.bbox,
                class_animal=ident.class_id,
                id_animal=ident.species,
                probability=ident.probability,
                mask=None,
                is_cetacean=True,
                cetacean_score=round(gate.positive_score, 4),
                rejected=True,
                rejection_reason=RejectionReason.LOW_CONFIDENCE.value,
                model_version=self.model_version,
            )

        # Stage 4: optional background mask
        mask_b64: str | None = None
        if generate_mask and img_bytes is not None:
            try:
                mask_b64 = self.identification.background_mask(img_bytes)
            except Exception as e:  # noqa: BLE001
                logger.warning("Background mask generation failed: %s", e)

        return Detection(
            image_ind=filename,
            bbox=ident.bbox,
            class_animal=ident.class_id,
            id_animal=ident.species,
            probability=ident.probability,
            mask=mask_b64,
            is_cetacean=True,
            cetacean_score=round(gate.positive_score, 4),
            rejected=False,
            rejection_reason=None,
            model_version=self.model_version,
        )
