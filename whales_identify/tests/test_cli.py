"""Smoke tests for the biologist CLI.

These avoid loading the heavy ML stack by patching the pipeline factory.
"""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def stub_pipeline(monkeypatch):
    from whales_identify import cli

    fake = MagicMock()
    fake.warmup = MagicMock()

    def _predict(pil_img, filename, img_bytes=None, generate_mask=True):
        from types import SimpleNamespace

        return SimpleNamespace(
            image_ind=filename,
            bbox=[0, 0, pil_img.width, pil_img.height],
            class_animal="1a71fbb72250",
            id_animal="humpback_whale",
            probability=0.92,
            mask=None,
            is_cetacean=True,
            cetacean_score=0.91,
            rejected=False,
            rejection_reason=None,
            model_version="stub-v1",
            model_dump=lambda: {
                "image_ind": filename,
                "bbox": [0, 0, pil_img.width, pil_img.height],
                "class_animal": "1a71fbb72250",
                "id_animal": "humpback_whale",
                "probability": 0.92,
                "mask": None,
                "is_cetacean": True,
                "cetacean_score": 0.91,
                "rejected": False,
                "rejection_reason": None,
                "model_version": "stub-v1",
            },
        )

    fake.predict = _predict
    fake.anti_fraud = types.SimpleNamespace(
        score=lambda pil: types.SimpleNamespace(
            positive_score=0.93,
            negative_score=0.07,
            is_cetacean=True,
            margin=0.86,
        )
    )

    monkeypatch.setattr(cli, "_get_pipeline", lambda: fake)
    return fake


@pytest.fixture
def temp_image(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    p = tmp_path / "test.jpg"
    Image.new("RGB", (50, 50), color=(0, 200, 0)).save(p)
    return p


def test_cli_predict_human(stub_pipeline, temp_image, capsys):
    from whales_identify.cli import main

    rc = main(["predict", str(temp_image)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "humpback_whale" in out


def test_cli_predict_json(stub_pipeline, temp_image, capsys):
    from whales_identify.cli import main

    rc = main(["predict", str(temp_image), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    import json

    data = json.loads(out)
    assert data["id_animal"] == "humpback_whale"
    assert data["rejected"] is False


def test_cli_batch_csv(stub_pipeline, tmp_path, capsys):
    pytest.importorskip("PIL")
    from PIL import Image

    src = tmp_path / "images"
    src.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), color=(0, 200, 0)).save(src / f"img{i}.jpg")
    out = tmp_path / "out.csv"

    from whales_identify.cli import main

    rc = main(["batch", str(src), "--csv", str(out)])
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    assert "humpback_whale" in text
    assert text.count("\n") >= 4  # header + 3 rows


def test_cli_verify_accepted(stub_pipeline, temp_image, capsys):
    from whales_identify.cli import main

    rc = main(["verify", str(temp_image)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ACCEPTED" in out
