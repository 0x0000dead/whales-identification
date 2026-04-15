"""Unit tests for ``EnsemblePipeline`` — §3.6 Комплексная CV-архитектура.

All three stages (CLIP gate, EfficientNet-B4 ArcFace, YOLOv8 bbox) are
replaced with ``MagicMock`` so the tests exercise the orchestration logic
without touching torch, open_clip, or ultralytics. This keeps the suite fast
(< 1 s) and GPU-free — a hard requirement for CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from whales_be_service.inference.ensemble import (
    EnsembleConfig,
    EnsemblePipeline,
    YoloV8BboxStub,
    build_ensemble_from_config,
)
from whales_be_service.inference.schemas import (
    GateResult,
    PredictionResult,
    RejectionReason,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_gate(is_cetacean: bool = True, pos: float = 0.9) -> MagicMock:
    gate = MagicMock()
    gate.score.return_value = GateResult(
        positive_score=pos,
        negative_score=1 - pos,
        is_cetacean=is_cetacean,
        margin=2 * pos - 1,
    )
    gate._load = MagicMock()
    return gate


def _make_ident(
    probability: float = 0.9,
    raise_on_predict: bool = False,
    version: str = "effb4-arcface-v1",
) -> MagicMock:
    ident = MagicMock()
    ident.model_version = version

    def _predict(pil_img: Image.Image) -> PredictionResult:
        if raise_on_predict:
            raise FileNotFoundError("effb4 weights missing")
        return PredictionResult(
            class_id="1a71fbb72250",
            species="humpback_whale",
            probability=probability,
            bbox=[0, 0, pil_img.width, pil_img.height],
        )

    ident.predict.side_effect = _predict
    ident.background_mask.return_value = "b64_mask"
    ident._load = MagicMock()
    return ident


def _make_yolo(bbox: list[int] | None = None) -> MagicMock:
    yolo = MagicMock()
    yolo.detect.return_value = bbox if bbox is not None else [10, 20, 100, 150]
    yolo._load = MagicMock()
    return yolo


@pytest.fixture
def img() -> Image.Image:
    return Image.new("RGB", (200, 200), color=(32, 64, 128))


# ---------------------------------------------------------------------------
# Stage-by-stage behaviour
# ---------------------------------------------------------------------------


class TestEnsembleBranching:
    def test_accepts_cetacean_and_uses_yolo_bbox(self, img: Image.Image) -> None:
        gate = _make_gate(is_cetacean=True, pos=0.91)
        ident = _make_ident(probability=0.88)
        yolo = _make_yolo(bbox=[5, 6, 150, 170])
        pipe = EnsemblePipeline(
            gate,
            ident,
            bbox_detector=yolo,
            config=EnsembleConfig(
                active_stages=["clip_gate", "effb4_arcface", "yolov8_bbox"],
                min_confidence=0.05,
            ),
        )

        det = pipe.predict(img, "whale.jpg", img_bytes=b"\x00", generate_mask=False)

        assert det.rejected is False
        assert det.is_cetacean is True
        assert det.cetacean_score == 0.91
        assert det.class_animal == "1a71fbb72250"
        assert det.id_animal == "humpback_whale"
        # YOLOv8 bbox must override the identification stage's full-image bbox.
        assert det.bbox == [5, 6, 150, 170]
        yolo.detect.assert_called_once()

    def test_gate_rejects_short_circuits_ident(self, img: Image.Image) -> None:
        gate = _make_gate(is_cetacean=False, pos=0.08)
        ident = _make_ident()
        yolo = _make_yolo()
        pipe = EnsemblePipeline(
            gate,
            ident,
            bbox_detector=yolo,
            config=EnsembleConfig(
                active_stages=["clip_gate", "effb4_arcface", "yolov8_bbox"],
            ),
        )

        det = pipe.predict(img, "dog.png")

        assert det.rejected is True
        assert det.rejection_reason == RejectionReason.NOT_A_MARINE_MAMMAL.value
        assert det.is_cetacean is False
        assert det.cetacean_score == 0.08
        ident.predict.assert_not_called()
        yolo.detect.assert_not_called()

    def test_low_confidence_is_rejected(self, img: Image.Image) -> None:
        gate = _make_gate(is_cetacean=True)
        ident = _make_ident(probability=0.01)
        pipe = EnsemblePipeline(
            gate,
            ident,
            config=EnsembleConfig(
                active_stages=["clip_gate", "effb4_arcface"],
                min_confidence=0.10,
            ),
        )

        det = pipe.predict(img, "blur.jpg")
        assert det.rejected is True
        assert det.rejection_reason == RejectionReason.LOW_CONFIDENCE.value
        assert det.probability == 0.01

    def test_identification_weights_missing_fallback(self, img: Image.Image) -> None:
        gate = _make_gate(is_cetacean=True, pos=0.80)
        ident = _make_ident(raise_on_predict=True)
        pipe = EnsemblePipeline(
            gate,
            ident,
            config=EnsembleConfig(active_stages=["clip_gate", "effb4_arcface"]),
        )

        det = pipe.predict(img, "x.jpg")
        assert det.rejected is False
        assert det.is_cetacean is True
        assert det.id_animal == "cetacean_unidentified"
        assert det.probability == 0.80

    def test_yolo_disabled_keeps_identification_bbox(self, img: Image.Image) -> None:
        gate = _make_gate(is_cetacean=True)
        ident = _make_ident(probability=0.85)
        yolo = _make_yolo(bbox=[1, 2, 3, 4])
        pipe = EnsemblePipeline(
            gate,
            ident,
            bbox_detector=yolo,
            config=EnsembleConfig(active_stages=["clip_gate", "effb4_arcface"]),
        )

        det = pipe.predict(img, "x.jpg")
        assert det.bbox == [0, 0, img.width, img.height]
        yolo.detect.assert_not_called()

    def test_clip_disabled_runs_identification_only(self, img: Image.Image) -> None:
        gate = _make_gate(is_cetacean=False)  # would reject if active
        ident = _make_ident(probability=0.9)
        pipe = EnsemblePipeline(
            gate,
            ident,
            config=EnsembleConfig(active_stages=["effb4_arcface"]),
        )

        det = pipe.predict(img, "x.jpg")
        # Gate is disabled → cetacean_score stays at default 1.0
        assert det.cetacean_score == 1.0
        assert det.rejected is False
        assert det.class_animal == "1a71fbb72250"
        gate.score.assert_not_called()

    def test_ident_stage_disabled_returns_gate_only(self, img: Image.Image) -> None:
        gate = _make_gate(is_cetacean=True, pos=0.77)
        ident = _make_ident()
        pipe = EnsemblePipeline(
            gate,
            ident,
            config=EnsembleConfig(active_stages=["clip_gate"]),
        )

        det = pipe.predict(img, "x.jpg")
        assert det.id_animal == "cetacean_unidentified"
        assert det.probability == 0.77
        ident.predict.assert_not_called()

    def test_yolo_failure_falls_back_to_ident_bbox(self, img: Image.Image) -> None:
        gate = _make_gate(is_cetacean=True)
        ident = _make_ident(probability=0.9)
        yolo = MagicMock()
        yolo.detect.side_effect = RuntimeError("CUDA OOM")
        pipe = EnsemblePipeline(
            gate,
            ident,
            bbox_detector=yolo,
            config=EnsembleConfig(
                active_stages=["clip_gate", "effb4_arcface", "yolov8_bbox"],
            ),
        )

        det = pipe.predict(img, "x.jpg")
        assert det.rejected is False
        assert det.bbox == [0, 0, img.width, img.height]


# ---------------------------------------------------------------------------
# Metadata & config
# ---------------------------------------------------------------------------


class TestEnsembleMetadata:
    def test_model_version_embeds_stages(self, img: Image.Image) -> None:
        gate = _make_gate()
        ident = _make_ident(version="effb4-arcface-v1")
        pipe = EnsemblePipeline(
            gate,
            ident,
            config=EnsembleConfig(active_stages=["clip_gate", "effb4_arcface"]),
        )
        assert "effb4-arcface-v1" in pipe.model_version
        assert "clip_gate" in pipe.model_version
        assert "effb4_arcface" in pipe.model_version
        assert "ensemble" in pipe.model_version

    def test_warmup_swallows_stage_errors(self) -> None:
        gate = _make_gate()
        gate._load.side_effect = RuntimeError("torch missing")
        ident = _make_ident()
        ident._load.side_effect = RuntimeError("weights missing")
        pipe = EnsemblePipeline(
            gate,
            ident,
            config=EnsembleConfig(active_stages=["clip_gate", "effb4_arcface"]),
        )
        pipe.warmup()  # must not raise

    def test_unknown_stage_is_ignored(self, img: Image.Image) -> None:
        gate = _make_gate()
        ident = _make_ident(probability=0.9)
        pipe = EnsemblePipeline(
            gate,
            ident,
            config=EnsembleConfig(
                active_stages=["clip_gate", "effb4_arcface", "nonexistent_stage"],
            ),
        )
        # Warmup must not crash on unknown stage name.
        pipe.warmup()
        det = pipe.predict(img, "x.jpg")
        assert det.rejected is False


class TestBuildFromConfig:
    def test_parses_yaml_block(self) -> None:
        cfg_dict = {
            "mode": "ensemble",
            "stages": ["clip_gate", "effb4_arcface", "yolov8_bbox"],
            "active_stages": ["clip_gate", "effb4_arcface"],
            "min_confidence": 0.07,
        }
        cfg = build_ensemble_from_config(cfg_dict)
        assert cfg.active_stages == ["clip_gate", "effb4_arcface"]
        assert cfg.min_confidence == 0.07

    def test_defaults_when_empty(self) -> None:
        cfg = build_ensemble_from_config({})
        assert cfg.active_stages == ["clip_gate", "effb4_arcface"]
        assert cfg.min_confidence == 0.05

    def test_falls_back_to_stages_when_active_missing(self) -> None:
        cfg = build_ensemble_from_config(
            {"stages": ["clip_gate", "effb4_arcface", "yolov8_bbox"]}
        )
        assert "yolov8_bbox" in cfg.active_stages


class TestYoloStub:
    def test_stub_returns_full_image_bbox(self, img: Image.Image) -> None:
        stub = YoloV8BboxStub()
        bbox = stub.detect(img)
        assert bbox == [0, 0, img.width, img.height]
