"""Ensemble inference pipeline: CLIP gate → EffB4 ArcFace → YOLOv8 bbox.

This module implements the §3.6 «Комплексная CV-архитектура» requirement:
a multi-stage pipeline that chains several specialised models to trade a
bit of latency for higher precision on high-stakes identifications.

The three stages are:

    stage 1  clip_gate        anti-fraud (reject non-cetacean imagery)
    stage 2  effb4_arcface    individual identification (13 837 classes)
    stage 3  yolov8_bbox      bbox refinement + crop for re-scoring

Stages can be individually enabled/disabled via ``models_config.yaml``:

.. code-block:: yaml

    models:
      ensemble:
        mode: ensemble
        stages: [clip_gate, effb4_arcface, yolov8_bbox]
        active_stages: [clip_gate, effb4_arcface]

The ensemble is an OPT-IN pipeline — the default `active_model` is still
`effb4_arcface`. To enable, set ``active_model: ensemble``. The existing
``InferencePipeline`` is NOT modified; ``EnsemblePipeline`` is a drop-in
replacement that exposes the same ``predict()`` contract used by the API
layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .schemas import GateResult, PredictionResult, RejectionReason

if TYPE_CHECKING:
    from PIL import Image

    from ..response_models import Detection
    from .anti_fraud import AntiFraudGate
    from .identification import IdentificationModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage protocols — lets us plug real models OR test doubles with zero glue.
# ---------------------------------------------------------------------------


class _GateStage(Protocol):
    def score(self, pil_img: Image.Image) -> GateResult: ...

    def _load(self) -> None: ...


class _IdentStage(Protocol):
    model_version: str

    def predict(self, pil_img: Image.Image) -> PredictionResult: ...

    def _load(self) -> None: ...


class _BboxStage(Protocol):
    """Optional third stage — YOLOv8 (or any detector returning xywh bbox)."""

    def detect(
        self, pil_img: Image.Image
    ) -> list[int] | None: ...  # [x, y, w, h] or None


# ---------------------------------------------------------------------------
# Stub YOLOv8 detector — real wiring happens in a follow-up. We ship a stub
# so the ensemble code path is testable today without a 300 MB weights download.
# ---------------------------------------------------------------------------


class YoloV8BboxStub:
    """Placeholder for the real YOLOv8 detector.

    Behaviour: returns the full-image bbox. This is intentionally a no-op so
    that the ensemble produces identical outputs to the single-model pipeline
    when the detector is stubbed — all the downstream code (API, drift
    monitor, webhooks) keeps working unchanged.

    TODO: wire the real ultralytics.YOLO('yolov8n.pt') model once weights
    are published to the EcoMarineAI HF org. Expected interface:

        from ultralytics import YOLO
        self._model = YOLO("yolov8n-whales.pt")
        result = self._model(np.array(pil_img))[0]
        box = result.boxes.xywh[0].tolist()  # [cx, cy, w, h]
        return [int(box[0] - box[2]/2), int(box[1] - box[3]/2),
                int(box[2]), int(box[3])]
    """

    model_version: str = "yolov8-stub-v0"

    def detect(self, pil_img: Image.Image) -> list[int] | None:
        # Full-image bbox — downstream code treats this as "no refinement".
        return [0, 0, pil_img.width, pil_img.height]

    def _load(self) -> None:  # pragma: no cover - stub
        return None


# ---------------------------------------------------------------------------
# Ensemble pipeline.
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    """Which stages to run. Unknown names are ignored with a warning."""

    active_stages: list[str] = field(
        default_factory=lambda: ["clip_gate", "effb4_arcface"]
    )
    min_confidence: float = 0.05


class EnsemblePipeline:
    """Multi-stage cetacean identifier — drop-in replacement for
    ``InferencePipeline`` when ``active_model: ensemble`` in ``models_config.yaml``.

    Contract matches ``InferencePipeline.predict()`` so ``main.py`` and the
    API layer don't need to know which pipeline is wired — they both return
    a ``Detection``.
    """

    def __init__(
        self,
        anti_fraud: AntiFraudGate | _GateStage,
        identification: IdentificationModel | _IdentStage,
        bbox_detector: _BboxStage | None = None,
        config: EnsembleConfig | None = None,
    ) -> None:
        self.anti_fraud = anti_fraud
        self.identification = identification
        self.bbox_detector = bbox_detector or YoloV8BboxStub()
        self.config = config or EnsembleConfig()

    # ------------------------------------------------------------------ API

    @property
    def model_version(self) -> str:
        """Composite version: <ident>+ensemble(<stages>)."""
        base = getattr(self.identification, "model_version", "unknown")
        stages = "+".join(self.config.active_stages)
        return f"{base}+ensemble({stages})"

    def warmup(self) -> None:
        """Force lazy-load on each enabled stage. Swallows per-stage errors so
        one missing model (e.g. YOLOv8 weights absent) does NOT block startup.
        """
        for stage_name, stage in self._iter_stages():
            loader = getattr(stage, "_load", None)
            if loader is None:
                continue
            try:
                loader()
                logger.info("Ensemble stage %s warmed up.", stage_name)
            except Exception as e:  # noqa: BLE001
                logger.warning("Ensemble stage %s warmup failed: %s", stage_name, e)

    def predict(
        self,
        pil_img: Image.Image,
        filename: str,
        img_bytes: bytes | None = None,
        generate_mask: bool = True,
    ) -> Detection:
        """Run every active stage in order. Rejection is short-circuited the
        moment a stage opts out (gate says "not a cetacean", or identification
        confidence < min_confidence).
        """
        from ..response_models import Detection  # noqa: PLC0415

        cetacean_score = 1.0  # default when clip_gate is disabled
        gate_result: GateResult | None = None

        # ------------------------------------------------------------------
        # Stage 1: CLIP anti-fraud gate
        # ------------------------------------------------------------------
        if "clip_gate" in self.config.active_stages:
            gate_result = self.anti_fraud.score(pil_img)
            cetacean_score = round(gate_result.positive_score, 4)
            if not gate_result.is_cetacean:
                logger.info(
                    "Ensemble: CLIP gate rejected '%s' (pos=%.3f)",
                    filename,
                    gate_result.positive_score,
                )
                return Detection(
                    image_ind=filename,
                    bbox=[0, 0, pil_img.width, pil_img.height],
                    class_animal="",
                    id_animal="unknown",
                    probability=0.0,
                    mask=None,
                    is_cetacean=False,
                    cetacean_score=cetacean_score,
                    rejected=True,
                    rejection_reason=RejectionReason.NOT_A_MARINE_MAMMAL.value,
                    model_version=self.model_version,
                )

        # ------------------------------------------------------------------
        # Stage 2: identification (EfficientNet-B4 ArcFace)
        # ------------------------------------------------------------------
        if "effb4_arcface" not in self.config.active_stages:
            # No identification stage — surface as "gate-only" verdict.
            return Detection(
                image_ind=filename,
                bbox=[0, 0, pil_img.width, pil_img.height],
                class_animal="",
                id_animal="cetacean_unidentified",
                probability=cetacean_score,
                mask=None,
                is_cetacean=True,
                cetacean_score=cetacean_score,
                rejected=False,
                rejection_reason=None,
                model_version=self.model_version,
            )

        try:
            ident = self.identification.predict(pil_img)
        except FileNotFoundError as e:
            logger.warning(
                "Ensemble: identification weights missing (%s); gate-only result.", e
            )
            return Detection(
                image_ind=filename,
                bbox=[0, 0, pil_img.width, pil_img.height],
                class_animal="",
                id_animal="cetacean_unidentified",
                probability=cetacean_score,
                mask=None,
                is_cetacean=True,
                cetacean_score=cetacean_score,
                rejected=False,
                rejection_reason=None,
                model_version=self.model_version,
            )

        if ident.probability < self.config.min_confidence:
            return Detection(
                image_ind=filename,
                bbox=ident.bbox,
                class_animal=ident.class_id,
                id_animal=ident.species,
                probability=ident.probability,
                mask=None,
                is_cetacean=True,
                cetacean_score=cetacean_score,
                rejected=True,
                rejection_reason=RejectionReason.LOW_CONFIDENCE.value,
                model_version=self.model_version,
            )

        # ------------------------------------------------------------------
        # Stage 3: YOLOv8 bbox refinement (optional).
        # ------------------------------------------------------------------
        bbox = ident.bbox
        if "yolov8_bbox" in self.config.active_stages:
            try:
                refined = self.bbox_detector.detect(pil_img)
                if refined is not None:
                    bbox = refined
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Ensemble: YOLOv8 detect failed (%s); keep ident bbox.", e
                )

        # ------------------------------------------------------------------
        # Optional background mask
        # ------------------------------------------------------------------
        mask_b64: str | None = None
        if generate_mask and img_bytes is not None:
            bg_fn = getattr(self.identification, "background_mask", None)
            if bg_fn is not None:
                try:
                    mask_b64 = bg_fn(img_bytes)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Ensemble: background mask failed (%s).", e)

        return Detection(
            image_ind=filename,
            bbox=bbox,
            class_animal=ident.class_id,
            id_animal=ident.species,
            probability=ident.probability,
            mask=mask_b64,
            is_cetacean=True,
            cetacean_score=cetacean_score,
            rejected=False,
            rejection_reason=None,
            model_version=self.model_version,
        )

    # -------------------------------------------------------------- helpers

    def _iter_stages(self):
        """Yield (name, stage_object) pairs for every *active* stage."""
        mapping = {
            "clip_gate": self.anti_fraud,
            "effb4_arcface": self.identification,
            "yolov8_bbox": self.bbox_detector,
        }
        for name in self.config.active_stages:
            stage = mapping.get(name)
            if stage is None:
                logger.warning("Ensemble: unknown stage '%s' — skipping.", name)
                continue
            yield name, stage


# ---------------------------------------------------------------------------
# Factory helpers.
# ---------------------------------------------------------------------------


def build_ensemble_from_config(config_dict: dict) -> EnsembleConfig:
    """Translate the ``ensemble`` block from ``models_config.yaml`` into
    an ``EnsembleConfig``. Unknown keys are ignored (forward-compat).
    """
    stages = list(config_dict.get("active_stages") or config_dict.get("stages") or [])
    if not stages:
        stages = ["clip_gate", "effb4_arcface"]
    return EnsembleConfig(
        active_stages=stages,
        min_confidence=float(config_dict.get("min_confidence", 0.05)),
    )
