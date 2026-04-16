"""Unit tests for the ``InferencePipeline`` orchestrator.

We stub both stages (anti-fraud gate and identification) so the test exercises
the orchestration logic without touching torch or CLIP.
"""

from unittest.mock import MagicMock

import pytest
from PIL import Image

from whales_be_service.inference.pipeline import InferencePipeline
from whales_be_service.inference.schemas import (
    GateResult,
    RejectionReason,
)


def _make_gate(is_cetacean: bool, pos: float = 0.85):
    g = MagicMock()
    g.score.return_value = GateResult(
        positive_score=pos,
        negative_score=1 - pos,
        is_cetacean=is_cetacean,
        margin=2 * pos - 1,
    )
    g._load = MagicMock()
    return g


_TOPK_FIXTURE = [
    ("1a71fbb72250", "humpback_whale", 0.90),
    ("abc456def789", "killer_whale", 0.54),
    ("cafe0987ba54", "bottlenose_dolphin", 0.27),
    ("dead01234567", "fin_whale", 0.09),
    ("beef89abcdef", "blue_whale", 0.04),
]


def _make_ident(
    probability: float = 0.9,
    raise_on_load: bool = False,
    raise_on_predict: bool = False,
):
    """Return a mock IdentificationModel that uses predict_topk (pipeline API)."""
    i = MagicMock()
    i.model_version = "stub-effb4-v1"

    def _load():
        if raise_on_load:
            raise FileNotFoundError("weights missing")

    i._load.side_effect = _load

    def _predict_topk(_pil, k: int = 5):
        if raise_on_predict:
            raise FileNotFoundError("weights missing")
        # Return k-length list of (class_id, species, prob) tuples.
        scaled = [
            (cls, sp, round(prob * probability / 0.90, 4))
            for cls, sp, prob in _TOPK_FIXTURE[:k]
        ]
        return scaled

    i.predict_topk.side_effect = _predict_topk
    i.background_mask.return_value = "base64_mask_placeholder"
    return i


class TestInferencePipelineBranching:
    @pytest.fixture
    def img(self):
        return Image.new("RGB", (100, 100), color=(0, 128, 200))

    def test_gate_accepts_and_identification_succeeds(self, img):
        gate = _make_gate(is_cetacean=True, pos=0.9)
        ident = _make_ident(probability=0.88)
        pipe = InferencePipeline(gate, ident, min_confidence=0.1)

        det = pipe.predict(img, "whale.jpg", img_bytes=b"\x00", generate_mask=False)
        assert det.rejected is False
        assert det.rejection_reason is None
        assert det.is_cetacean is True
        assert det.cetacean_score == 0.9
        assert det.class_animal == "1a71fbb72250"
        assert det.id_animal == "humpback_whale"

    def test_gate_rejects_returns_not_marine_mammal(self, img):
        gate = _make_gate(is_cetacean=False, pos=0.12)
        ident = _make_ident(probability=0.88)
        pipe = InferencePipeline(gate, ident, min_confidence=0.1)

        det = pipe.predict(img, "text.png")
        assert det.rejected is True
        assert det.rejection_reason == RejectionReason.NOT_A_MARINE_MAMMAL.value
        assert det.is_cetacean is False
        assert det.cetacean_score == 0.12
        assert det.class_animal == ""
        # Identification must not be called when the gate rejects
        ident.predict_topk.assert_not_called()

    def test_low_confidence_branch(self, img):
        gate = _make_gate(is_cetacean=True, pos=0.9)
        ident = _make_ident(probability=0.01)
        pipe = InferencePipeline(gate, ident, min_confidence=0.1)

        det = pipe.predict(img, "low.jpg")
        assert det.rejected is True
        assert det.rejection_reason == RejectionReason.LOW_CONFIDENCE.value
        assert det.is_cetacean is True
        assert det.probability == pytest.approx(0.01, abs=1e-4)

    def test_candidates_populated_on_accepted_prediction(self, img):
        """top-K prediction: top-1 → main result, top-2..5 → candidates list."""
        gate = _make_gate(is_cetacean=True, pos=0.9)
        ident = _make_ident(probability=0.9)
        pipe = InferencePipeline(gate, ident, min_confidence=0.1)

        det = pipe.predict(img, "whale.jpg")
        assert len(det.candidates) == 4, "top-2..5 must appear as 4 candidates"
        first_cand = det.candidates[0]
        assert first_cand.id_animal == "killer_whale"
        assert 0.0 <= first_cand.probability <= 1.0

    def test_low_confidence_includes_candidates(self, img):
        """Even when low-confidence rejected, candidates list is populated."""
        gate = _make_gate(is_cetacean=True, pos=0.9)
        ident = _make_ident(probability=0.01)
        pipe = InferencePipeline(gate, ident, min_confidence=0.1)

        det = pipe.predict(img, "low.jpg")
        assert det.rejected is True
        # Candidates are still populated so the UI can show alternatives
        assert len(det.candidates) == 4

    def test_identification_weights_missing_fallback(self, img):
        gate = _make_gate(is_cetacean=True, pos=0.88)
        ident = _make_ident(raise_on_predict=True)
        pipe = InferencePipeline(gate, ident, min_confidence=0.1)

        det = pipe.predict(img, "fallback.jpg")
        assert det.rejected is False
        assert det.is_cetacean is True
        assert det.id_animal == "cetacean_unidentified"

    def test_warmup_does_not_crash_when_gate_fails(self, img):
        gate = _make_gate(is_cetacean=True)
        gate._load.side_effect = RuntimeError("broken")
        ident = _make_ident()
        pipe = InferencePipeline(gate, ident)
        pipe.warmup()  # should just log, not raise

    def test_warmup_does_not_crash_when_ident_fails(self, img):
        gate = _make_gate(is_cetacean=True)
        ident = _make_ident(raise_on_load=True)
        pipe = InferencePipeline(gate, ident)
        pipe.warmup()

    def test_model_version_propagates(self, img):
        gate = _make_gate(is_cetacean=True)
        ident = _make_ident()
        pipe = InferencePipeline(gate, ident, min_confidence=0.01)
        det = pipe.predict(img, "x.jpg")
        assert det.model_version == "stub-effb4-v1"

    def test_batch_mode_skips_mask(self, img):
        gate = _make_gate(is_cetacean=True)
        ident = _make_ident()
        pipe = InferencePipeline(gate, ident, min_confidence=0.01)
        det = pipe.predict(img, "x.jpg", img_bytes=b"\x00", generate_mask=False)
        ident.background_mask.assert_not_called()
        assert det.mask is None
