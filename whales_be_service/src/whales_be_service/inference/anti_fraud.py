"""CLIP-based zero-shot anti-fraud gate.

Rejects images that are not photographs of whales or dolphins by computing
cosine similarity between the image embedding and a fixed set of positive vs
negative text prompts (see ``prompts.py``). The threshold is read from
``configs/anti_fraud_threshold.yaml`` (calibrated by
``scripts/calibrate_clip_threshold.py``); falls back to a sensible default if
the calibration file is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .prompts import ALL_PROMPTS, NUM_POSITIVE
from .schemas import GateResult

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _BASE_DIR / "configs" / "anti_fraud_threshold.yaml"
_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
_DEFAULT_THRESHOLD = 0.55


def _load_calibrated_threshold() -> float:
    if not _CONFIG_PATH.exists():
        logger.info(
            "Anti-fraud calibration file %s not found; using default threshold %.2f",
            _CONFIG_PATH,
            _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    threshold = float(data.get("threshold", _DEFAULT_THRESHOLD))
    logger.info(
        "Loaded calibrated anti-fraud threshold %.4f from %s",
        threshold,
        _CONFIG_PATH.name,
    )
    return threshold


class AntiFraudGate:
    """CLIP zero-shot gate. Lazy-loaded; safe to import without torch present."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        pretrained: str = _DEFAULT_PRETRAINED,
        threshold: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.threshold = (
            threshold if threshold is not None else _load_calibrated_threshold()
        )

        self._loaded = False
        self._unavailable = False  # True if open_clip can't be imported (degraded mode)
        self._model = None
        self._preprocess = None
        self._text_features = None
        self._device = None

    def _load(self) -> None:
        if self._loaded or self._unavailable:
            return

        try:
            import open_clip  # noqa: PLC0415
            import torch  # noqa: PLC0415
        except ImportError as e:
            self._unavailable = True
            logger.error(
                "open_clip_torch not available — anti-fraud gate will operate in "
                "DEGRADED (permissive) mode. Install with `poetry add open-clip-torch` "
                "to enable real specificity gating. Reason: %s",
                e,
            )
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self._device,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(self.model_name)

        text_tokens = tokenizer(list(ALL_PROMPTS)).to(self._device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        self._model = model
        self._preprocess = preprocess
        self._text_features = text_features
        self._loaded = True
        logger.info(
            "Loaded AntiFraudGate: %s/%s, %d prompts (%d positive), threshold=%.4f",
            self.model_name,
            self.pretrained,
            len(ALL_PROMPTS),
            NUM_POSITIVE,
            self.threshold,
        )

    def score(self, pil_img: "Image.Image") -> GateResult:
        """Compute (positive_score, negative_score, is_cetacean, margin).

        In degraded mode (open_clip not installed), returns a permissive
        result that lets every image through — visible to operators via the
        `cetacean_score=0.5` sentinel and the WARNING logged by ``_load()``.
        """
        self._load()

        if self._unavailable:
            return GateResult(
                positive_score=0.5,
                negative_score=0.5,
                is_cetacean=True,  # permissive — let identification handle it
                margin=0.0,
            )

        import torch  # noqa: PLC0415

        img = self._preprocess(pil_img.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sims = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)[0]

        pos = float(sims[:NUM_POSITIVE].sum().item())
        neg = float(sims[NUM_POSITIVE:].sum().item())
        return GateResult(
            positive_score=pos,
            negative_score=neg,
            is_cetacean=pos >= self.threshold,
            margin=pos - neg,
        )

    def score_many(self, pil_imgs: list) -> list[GateResult]:
        """Batched scoring — useful for calibration/benchmark scripts."""
        self._load()

        if self._unavailable:
            return [
                GateResult(
                    positive_score=0.5, negative_score=0.5, is_cetacean=True, margin=0.0
                )
                for _ in pil_imgs
            ]

        import torch  # noqa: PLC0415

        if not pil_imgs:
            return []
        batch = torch.stack(
            [self._preprocess(img.convert("RGB")) for img in pil_imgs]
        ).to(self._device)
        with torch.no_grad():
            image_features = self._model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sims = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)

        results: list[GateResult] = []
        for row in sims:
            pos = float(row[:NUM_POSITIVE].sum().item())
            neg = float(row[NUM_POSITIVE:].sum().item())
            results.append(
                GateResult(
                    positive_score=pos,
                    negative_score=neg,
                    is_cetacean=pos >= self.threshold,
                    margin=pos - neg,
                )
            )
        return results
