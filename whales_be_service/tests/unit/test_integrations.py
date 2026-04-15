"""Unit tests for integrations/*.py — sqlite_sink, postgres_sink, otel_sink.

Postgres and OTel tests use mocks so they never touch a real server.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _fake_detection(filename: str, rejected: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        image_ind=filename,
        bbox=[0, 0, 100, 80],
        class_animal="1a71fbb72250" if not rejected else "",
        id_animal="humpback_whale" if not rejected else "unknown",
        probability=0.92 if not rejected else 0.0,
        mask=None,
        is_cetacean=not rejected,
        cetacean_score=0.95 if not rejected else 0.1,
        rejected=rejected,
        rejection_reason=None if not rejected else "not_a_marine_mammal",
        model_version="stub-v1",
    )


@pytest.fixture
def stub_pipeline():
    """Return a MagicMock that looks like InferencePipeline.predict()."""
    p = MagicMock()
    p.warmup = MagicMock()

    def _predict(pil_img=None, filename="x", img_bytes=None, generate_mask=False):
        return _fake_detection(filename, rejected=(filename.startswith("neg")))

    p.predict.side_effect = _predict
    return p


@pytest.fixture
def sample_dir(tmp_path):
    """Three stub images (2 positives, 1 negative)."""
    from PIL import Image

    for name in ["pos_a.jpg", "pos_b.jpg", "neg_a.jpg"]:
        Image.new("RGB", (50, 50), color=(0, 200, 0)).save(tmp_path / name)
    return tmp_path


class TestSqliteSink:
    def test_writes_rows_to_sqlite(
        self, stub_pipeline, sample_dir, monkeypatch, tmp_path
    ):
        mod = _load_module("sqlite_sink", REPO_ROOT / "integrations" / "sqlite_sink.py")
        monkeypatch.setattr(mod, "get_pipeline", lambda: stub_pipeline, raising=False)

        # Patch get_pipeline at the module level since the function does
        # `from whales_be_service.inference import get_pipeline` at call time.
        import whales_be_service.inference as inf_mod

        monkeypatch.setattr(inf_mod, "get_pipeline", lambda: stub_pipeline)

        db_path = tmp_path / "obs.sqlite"
        rc = mod.run(directory=sample_dir, db_path=db_path)
        assert rc == 0
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        rows = list(
            conn.execute(
                "SELECT filename, rejected, id_animal, probability FROM detections"
            )
        )
        conn.close()
        assert len(rows) == 3
        filenames = {r[0] for r in rows}
        assert "pos_a.jpg" in filenames
        assert "neg_a.jpg" in filenames
        pos_rows = [r for r in rows if not r[1]]
        neg_rows = [r for r in rows if r[1]]
        assert len(pos_rows) == 2
        assert len(neg_rows) == 1
        assert all(r[2] == "humpback_whale" for r in pos_rows)

    def test_schema_has_expected_columns(
        self, stub_pipeline, sample_dir, monkeypatch, tmp_path
    ):
        mod = _load_module(
            "sqlite_sink_schema", REPO_ROOT / "integrations" / "sqlite_sink.py"
        )
        import whales_be_service.inference as inf_mod

        monkeypatch.setattr(inf_mod, "get_pipeline", lambda: stub_pipeline)
        db_path = tmp_path / "schema.sqlite"
        mod.run(directory=sample_dir, db_path=db_path)
        conn = sqlite3.connect(str(db_path))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(detections)")]
        conn.close()
        for expected in [
            "filename",
            "rejected",
            "rejection_reason",
            "is_cetacean",
            "cetacean_score",
            "class_animal",
            "id_animal",
            "probability",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "model_version",
        ]:
            assert expected in cols, f"missing column {expected}"


class TestPostgresSink:
    def test_raises_without_psycopg(self, stub_pipeline, sample_dir, monkeypatch):
        """When psycopg is not installed, the sink must exit cleanly with rc=2."""
        monkeypatch.setitem(sys.modules, "psycopg", None)
        mod = _load_module(
            "postgres_sink", REPO_ROOT / "integrations" / "postgres_sink.py"
        )
        with pytest.raises(SystemExit) as exc:
            mod.run(directory=sample_dir, dsn="postgresql://stub")
        assert exc.value.code == 2

    def test_insert_with_mocked_psycopg(self, stub_pipeline, sample_dir, monkeypatch):
        """When psycopg IS available, the sink should call execute for each image."""
        fake_cursor = MagicMock()
        fake_cursor.__enter__ = MagicMock(return_value=fake_cursor)
        fake_cursor.__exit__ = MagicMock(return_value=None)

        fake_conn = MagicMock()
        fake_conn.cursor = MagicMock(return_value=fake_cursor)
        fake_conn.__enter__ = MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = MagicMock(return_value=None)
        fake_conn.commit = MagicMock()

        fake_psycopg = MagicMock()
        fake_psycopg.connect = MagicMock(return_value=fake_conn)

        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
        import whales_be_service.inference as inf_mod

        monkeypatch.setattr(inf_mod, "get_pipeline", lambda: stub_pipeline)

        mod = _load_module(
            "postgres_sink2", REPO_ROOT / "integrations" / "postgres_sink.py"
        )
        rc = mod.run(directory=sample_dir, dsn="postgresql://stub")
        assert rc == 0
        # One schema create + 3 inserts (execute called)
        assert fake_cursor.execute.call_count >= 4
        # All three test images should have been INSERTed
        insert_calls = [
            c
            for c in fake_cursor.execute.call_args_list
            if "INSERT INTO detections" in (c[0][0] if c[0] else "")
        ]
        assert len(insert_calls) == 3


class TestOtelSinkNoopMode:
    def test_runs_without_opentelemetry_installed(
        self, stub_pipeline, sample_dir, monkeypatch
    ):
        """When opentelemetry isn't installed, the sink must still process
        images in no-op mode (the point is: service-side sink failure shouldn't
        stop predictions).
        """
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("opentelemetry"):
                monkeypatch.setitem(sys.modules, mod_name, None)

        import whales_be_service.inference as inf_mod

        monkeypatch.setattr(inf_mod, "get_pipeline", lambda: stub_pipeline)

        mod = _load_module("otel_sink", REPO_ROOT / "integrations" / "otel_sink.py")
        rc = mod.run(directory=sample_dir, endpoint=None, service_name="test")
        assert rc == 0
        # 3 predictions expected from 3 images in sample_dir
        assert stub_pipeline.predict.call_count == 3
