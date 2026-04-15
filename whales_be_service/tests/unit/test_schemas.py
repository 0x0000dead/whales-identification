"""Unit tests for inference schemas (dataclasses + enum)."""

import pytest

from whales_be_service.inference.schemas import (
    GateResult,
    PredictionResult,
    RejectionReason,
)


class TestRejectionReason:
    def test_values_are_stable_strings(self):
        assert RejectionReason.NOT_A_MARINE_MAMMAL.value == "not_a_marine_mammal"
        assert RejectionReason.LOW_CONFIDENCE.value == "low_confidence"
        assert RejectionReason.CORRUPTED_IMAGE.value == "corrupted_image"

    def test_str_subclass_interop(self):
        # The enum inherits from str so literals compare directly.
        assert RejectionReason.NOT_A_MARINE_MAMMAL == "not_a_marine_mammal"

    def test_roundtrip_via_value(self):
        v = RejectionReason.LOW_CONFIDENCE.value
        assert RejectionReason(v) is RejectionReason.LOW_CONFIDENCE


class TestGateResult:
    def test_positive(self):
        r = GateResult(
            positive_score=0.9, negative_score=0.1, is_cetacean=True, margin=0.8
        )
        assert r.is_cetacean is True
        assert r.positive_score == 0.9

    def test_immutable(self):
        r = GateResult(
            positive_score=0.5, negative_score=0.5, is_cetacean=True, margin=0
        )
        with pytest.raises(Exception):
            r.positive_score = 1.0  # frozen


class TestPredictionResult:
    def test_minimal_construction(self):
        r = PredictionResult(
            class_id="abc123",
            species="humpback_whale",
            probability=0.85,
            bbox=[0, 0, 100, 100],
        )
        assert r.class_id == "abc123"
        assert r.embedding is None

    def test_immutable(self):
        r = PredictionResult(
            class_id="x", species="y", probability=0.1, bbox=[0, 0, 1, 1]
        )
        with pytest.raises(Exception):
            r.class_id = "z"
