"""Unit tests for IdentificationModel's routing logic.

The heavy loads (torch + timm + 300MB checkpoint) are covered by the real
compute_metrics integration path; here we only exercise:
  * pre-flight file-existence checks raise FileNotFoundError before touching
    torch;
  * the backend selector picks effb4 > vit_full > resnet_fallback > error;
  * constructor defaults are sensible.
"""

from pathlib import Path

import pytest

from whales_be_service.inference.identification import IdentificationModel


class TestIdentificationModelRouting:
    def test_defaults_are_cheap(self):
        m = IdentificationModel()
        assert m._loaded is False
        assert m._mode == "uninitialised"
        assert m._model is None

    def test_all_missing_raises_file_not_found(self, tmp_path):
        m = IdentificationModel(
            csv_path=tmp_path / "nope.csv",
            ckpt_path=tmp_path / "nope.pt",
            fallback_ckpt=tmp_path / "nope2.pth",
            effb4_ckpt=tmp_path / "nope3.ckpt",
            effb4_classes=tmp_path / "nope4.npy",
            effb4_species_map=tmp_path / "nope5.csv",
        )
        with pytest.raises(FileNotFoundError) as exc:
            m._load()
        msg = str(exc.value)
        # The error message should name every path it tried so ops can see
        # which ones are missing.
        assert "effb4" in msg
        assert "ViT" in msg

    def test_effb4_takes_precedence_when_all_files_present(self, tmp_path, monkeypatch):
        """Simulate all three backends being present — effb4 should be chosen."""
        csv = tmp_path / "db.csv"
        ckpt = tmp_path / "m.pt"
        fallback = tmp_path / "r.pth"
        eff_ckpt = tmp_path / "e.ckpt"
        eff_cls = tmp_path / "classes.npy"
        eff_map = tmp_path / "map.csv"
        for p in (csv, ckpt, fallback, eff_ckpt, eff_cls, eff_map):
            p.write_bytes(b"stub")

        loaded_mode = []

        def fake_effb4(self):
            self._loaded = True
            self._mode = "effb4_15k"
            loaded_mode.append("effb4_15k")

        def fake_vit(self):
            self._loaded = True
            self._mode = "vit_full"
            loaded_mode.append("vit_full")

        def fake_resnet(self):
            self._loaded = True
            self._mode = "resnet_fallback"
            loaded_mode.append("resnet_fallback")

        monkeypatch.setattr(IdentificationModel, "_load_effb4_arcface", fake_effb4)
        monkeypatch.setattr(IdentificationModel, "_load_vit_full", fake_vit)
        monkeypatch.setattr(IdentificationModel, "_load_resnet_fallback", fake_resnet)

        m = IdentificationModel(
            csv_path=csv,
            ckpt_path=ckpt,
            fallback_ckpt=fallback,
            effb4_ckpt=eff_ckpt,
            effb4_classes=eff_cls,
            effb4_species_map=eff_map,
        )
        m._load()
        assert loaded_mode == ["effb4_15k"]

    def test_vit_picked_when_only_vit_present(self, tmp_path, monkeypatch):
        csv = tmp_path / "db.csv"
        ckpt = tmp_path / "m.pt"
        csv.write_bytes(b"stub")
        ckpt.write_bytes(b"stub")

        loaded = []

        def fake_vit(self):
            self._loaded = True
            self._mode = "vit_full"
            loaded.append("vit_full")

        monkeypatch.setattr(IdentificationModel, "_load_vit_full", fake_vit)

        m = IdentificationModel(
            csv_path=csv,
            ckpt_path=ckpt,
            fallback_ckpt=tmp_path / "nope.pth",
            effb4_ckpt=tmp_path / "nope.ckpt",
            effb4_classes=tmp_path / "nope.npy",
            effb4_species_map=tmp_path / "nope.csv",
        )
        m._load()
        assert loaded == ["vit_full"]

    def test_resnet_picked_when_only_resnet_present(self, tmp_path, monkeypatch):
        resnet = tmp_path / "r.pth"
        resnet.write_bytes(b"stub")

        loaded = []

        def fake_resnet(self):
            self._loaded = True
            self._mode = "resnet_fallback"
            loaded.append("resnet_fallback")

        monkeypatch.setattr(IdentificationModel, "_load_resnet_fallback", fake_resnet)

        m = IdentificationModel(
            csv_path=tmp_path / "nope.csv",
            ckpt_path=tmp_path / "nope.pt",
            fallback_ckpt=resnet,
            effb4_ckpt=tmp_path / "nope.ckpt",
            effb4_classes=tmp_path / "nope.npy",
            effb4_species_map=tmp_path / "nope.csv",
        )
        m._load()
        assert loaded == ["resnet_fallback"]
