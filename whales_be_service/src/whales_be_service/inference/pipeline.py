"""Inference pipeline orchestrator: anti-fraud gate → identification → response.

Public API: ``InferencePipeline.predict(pil_img, filename) -> Detection``.

This is the ONLY entry point used by ``main.py``. The previous design wired
random mock data directly in the route handler; that path has been removed.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from .anti_fraud import AntiFraudGate
from .identification import IdentificationModel
from .schemas import RejectionReason

if TYPE_CHECKING:
    from PIL import Image

    from ..response_models import Detection

logger = logging.getLogger(__name__)

_REPRODUCIBILITY_SEED = 2022


def set_deterministic_mode(seed: int = _REPRODUCIBILITY_SEED) -> None:
    """Fix all random seeds for fully deterministic inference.

    Call once at application startup (e.g. inside FastAPI lifespan), not inside
    class constructors — repeated calls reset global PRNG state and break test
    isolation between independent ``InferencePipeline`` instances.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Deterministic mode enabled (seed=%d).", seed)


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

    @property
    def model_version(self) -> str:
        """Reported in every Detection — reflects the actual loaded backend.
        Default for a fresh, un-warmed instance is ``effb4-arcface-v1``; the
        value is updated lazily by ``IdentificationModel._load()`` when the
        real backend is promoted.
        """
        return self.identification.model_version

    def warmup(self) -> None:
        """Force lazy-load of both stages. Called from FastAPI lifespan startup."""
        try:
            self.anti_fraud._load()  # noqa: SLF001
            logger.info("Anti-fraud gate warmed up.")
        except Exception as e:  # noqa: BLE001
            logger.warning("Anti-fraud gate warmup failed: %s", e)
        try:
            self.identification._load()  # noqa: SLF001
            logger.info(
                "Identification model warmed up (mode=%s, version=%s).",
                self.identification._mode,  # noqa: SLF001
                self.identification.model_version,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Identification model warmup failed (running without ID stage): %s",
                e,
            )

    def predict(
        self,
        pil_img: Image.Image,
        filename: str,
        img_bytes: bytes | None = None,
        generate_mask: bool = True,
    ) -> Detection:
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

        # Stage 2: identification — top-5 for candidates, top-1 for gating
        from ..response_models import Candidate  # noqa: PLC0415

        try:
            topk = self.identification.predict_topk(pil_img, k=5)
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

        # top-1 drives gating; remaining entries become alternative candidates.
        # bbox is always the full image (no separate object detector).
        top1_class, top1_species, top1_prob = topk[0]
        candidates = [
            Candidate(class_animal=c, id_animal=s, probability=p)
            for c, s, p in topk[1:]
        ]
        bbox = [0, 0, pil_img.width, pil_img.height]

        # Stage 3: confidence threshold gating
        if top1_prob < self.min_confidence:
            return Detection(
                image_ind=filename,
                bbox=bbox,
                class_animal=top1_class,
                id_animal=top1_species,
                probability=top1_prob,
                mask=None,
                is_cetacean=True,
                cetacean_score=round(gate.positive_score, 4),
                rejected=True,
                rejection_reason=RejectionReason.LOW_CONFIDENCE.value,
                model_version=self.model_version,
                candidates=candidates,
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
            bbox=bbox,
            class_animal=top1_class,
            id_animal=top1_species,
            probability=top1_prob,
            mask=mask_b64,
            is_cetacean=True,
            cetacean_score=round(gate.positive_score, 4),
            rejected=False,
            rejection_reason=None,
            model_version=self.model_version,
            candidates=candidates,
        )
