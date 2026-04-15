"""Unit tests for the CLIP anti-fraud gate.

We do NOT import ``open_clip`` here — the tests verify the gate's *shape* and
degradation behaviour, not its ML accuracy (that's covered by the integration
tests in ``tests/integration/test_metrics.py``).
"""

import sys

import pytest
from PIL import Image

from whales_be_service.inference.anti_fraud import (
    AntiFraudGate,
    _load_calibrated_threshold,
)
from whales_be_service.inference.prompts import (
    ALL_PROMPTS,
    NEGATIVE_PROMPTS,
    NUM_POSITIVE,
    POSITIVE_PROMPTS,
)


class TestPromptsInventory:
    def test_positive_and_negative_counts(self):
        assert len(POSITIVE_PROMPTS) >= 8
        assert len(NEGATIVE_PROMPTS) >= 10

    def test_all_prompts_is_sum(self):
        assert len(ALL_PROMPTS) == NUM_POSITIVE + len(NEGATIVE_PROMPTS)

    def test_num_positive_is_index_offset(self):
        assert ALL_PROMPTS[:NUM_POSITIVE] == POSITIVE_PROMPTS
        assert ALL_PROMPTS[NUM_POSITIVE:] == NEGATIVE_PROMPTS


class TestAntiFraudGateDegradedMode:
    """When ``open_clip`` cannot be imported, the gate must return permissive
    results and log a warning — never raise.
    """

    def test_import_error_sets_unavailable_flag(self, monkeypatch):
        # Shadow the real open_clip module with a placeholder that raises on
        # attribute access. Then call _load() — it should swallow the error.
        monkeypatch.setitem(sys.modules, "open_clip", None)
        gate = AntiFraudGate()
        gate._load()
        assert gate._unavailable is True
        assert gate._loaded is False

    def test_score_returns_permissive_in_degraded(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "open_clip", None)
        gate = AntiFraudGate()
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        res = gate.score(img)
        assert res.is_cetacean is True
        assert res.positive_score == 0.5
        assert res.negative_score == 0.5
        assert res.margin == 0.0

    def test_score_many_returns_permissive_in_degraded(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "open_clip", None)
        gate = AntiFraudGate()
        imgs = [Image.new("RGB", (10, 10)) for _ in range(3)]
        results = gate.score_many(imgs)
        assert len(results) == 3
        assert all(r.is_cetacean for r in results)


class TestCalibratedThresholdLoader:
    def test_fallback_default_when_file_missing(self, tmp_path, monkeypatch):
        from whales_be_service.inference import anti_fraud

        monkeypatch.setattr(anti_fraud, "_CONFIG_PATH", tmp_path / "nope.yaml")
        assert _load_calibrated_threshold() == pytest.approx(0.55)

    def test_reads_calibrated_value(self, tmp_path, monkeypatch):
        from whales_be_service.inference import anti_fraud

        cfg = tmp_path / "threshold.yaml"
        cfg.write_text("threshold: 0.42\ntpr: 0.9\ntnr: 0.92\n")
        monkeypatch.setattr(anti_fraud, "_CONFIG_PATH", cfg)
        assert _load_calibrated_threshold() == pytest.approx(0.42)
