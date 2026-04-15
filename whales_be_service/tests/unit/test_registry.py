"""Unit tests for the model registry loader."""

import json

from whales_be_service.inference import registry


class TestRegistryParsing:
    def test_list_models_returns_empty_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "_REGISTRY_PATH", tmp_path / "missing.json")
        assert registry.list_models() == []

    def test_list_models_reads_file(self, tmp_path, monkeypatch):
        p = tmp_path / "registry.json"
        p.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "models": [
                        {"name": "a", "version": "1.0.0"},
                        {"name": "b", "version": "2.0.0"},
                    ],
                }
            )
        )
        monkeypatch.setattr(registry, "_REGISTRY_PATH", p)
        models = registry.list_models()
        assert len(models) == 2
        assert {m["name"] for m in models} == {"a", "b"}

    def test_get_model_metadata_by_name(self, tmp_path, monkeypatch):
        p = tmp_path / "registry.json"
        p.write_text(
            json.dumps(
                {
                    "models": [
                        {"name": "x", "version": "1.0.0", "weights_url": "https://..."}
                    ]
                }
            )
        )
        monkeypatch.setattr(registry, "_REGISTRY_PATH", p)
        assert registry.get_model_metadata("x")["version"] == "1.0.0"
        assert registry.get_model_metadata("missing") is None

    def test_get_model_metadata_with_version(self, tmp_path, monkeypatch):
        p = tmp_path / "registry.json"
        p.write_text(
            json.dumps(
                {
                    "models": [
                        {"name": "x", "version": "1.0.0"},
                        {"name": "x", "version": "2.0.0"},
                    ]
                }
            )
        )
        monkeypatch.setattr(registry, "_REGISTRY_PATH", p)
        assert registry.get_model_metadata("x", "1.0.0")["version"] == "1.0.0"
        assert registry.get_model_metadata("x", "2.0.0")["version"] == "2.0.0"
        assert registry.get_model_metadata("x", "3.0.0") is None
