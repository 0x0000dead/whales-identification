"""API tests for Stage 3 webhook + export endpoints.

We import ``whales_be_service.routers`` BEFORE ``main.app`` usage so the
routers are attached and the prediction recorder is wrapped. Without this
explicit import the side-effect would not run for test collection.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import whales_be_service.routers  # noqa: F401 — triggers router registration
from whales_be_service.export import get_history_store
from whales_be_service.main import app
from whales_be_service.webhooks import get_webhook_registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state():
    get_webhook_registry().clear()
    get_history_store().clear()
    yield
    get_webhook_registry().clear()
    get_history_store().clear()


def _png(color=(0, 200, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color).save(buf, format="PNG")
    return buf.getvalue()


def _red_png() -> bytes:
    return _png(color=(255, 0, 0))


# --- /v1/webhook/register -------------------------------------------------


def test_register_webhook_happy_path():
    resp = client.post(
        "/v1/webhook/register",
        json={
            "url": "https://client.example.com/hooks/whales",
            "events": ["batch_completed"],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "registered"
    assert isinstance(body["webhook_id"], str) and len(body["webhook_id"]) >= 16
    assert body["events"] == ["batch_completed"]
    # Registry now holds one subscription
    assert len(get_webhook_registry().list_all()) == 1


def test_register_webhook_rejects_empty_events():
    resp = client.post(
        "/v1/webhook/register",
        json={"url": "https://example.com/hooks", "events": []},
    )
    assert resp.status_code == 422  # pydantic min_length=1


def test_register_webhook_rejects_unknown_event():
    resp = client.post(
        "/v1/webhook/register",
        json={"url": "https://example.com/hooks", "events": ["meteor_strike"]},
    )
    assert resp.status_code == 422


def test_register_webhook_rejects_invalid_url():
    resp = client.post(
        "/v1/webhook/register",
        json={"url": "not-a-url", "events": ["batch_completed"]},
    )
    assert resp.status_code == 422


def test_register_webhook_rejects_duplicate_events():
    resp = client.post(
        "/v1/webhook/register",
        json={
            "url": "https://example.com/hooks",
            "events": ["batch_completed", "batch_completed"],
        },
    )
    assert resp.status_code == 422


# --- /v1/webhooks ---------------------------------------------------------


def test_list_webhooks_empty():
    resp = client.get("/v1/webhooks")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"webhooks": [], "count": 0}


def test_list_webhooks_after_register():
    client.post(
        "/v1/webhook/register",
        json={
            "url": "https://a.example.com/h",
            "events": ["batch_completed"],
        },
    )
    client.post(
        "/v1/webhook/register",
        json={
            "url": "https://b.example.com/h",
            "events": ["prediction_rejected"],
        },
    )
    resp = client.get("/v1/webhooks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    urls = {w["url"] for w in body["webhooks"]}
    assert "https://a.example.com/h" in urls
    assert "https://b.example.com/h" in urls


# --- /v1/webhook/{id} DELETE ---------------------------------------------


def test_delete_webhook_happy_path():
    reg = client.post(
        "/v1/webhook/register",
        json={
            "url": "https://example.com/hooks",
            "events": ["batch_completed"],
        },
    )
    webhook_id = reg.json()["webhook_id"]

    resp = client.delete(f"/v1/webhook/{webhook_id}")
    assert resp.status_code == 204
    assert resp.content == b""
    assert len(get_webhook_registry().list_all()) == 0


def test_delete_webhook_not_found():
    resp = client.delete("/v1/webhook/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "webhook_id not found"


# --- /v1/export JSON ------------------------------------------------------


def test_export_json_empty():
    resp = client.get("/v1/export?format=json")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"count": 0, "since": None, "records": []}


def test_export_json_captures_predictions():
    # Run two predictions (one accepted, one rejected) through the pipeline —
    # the routers.py hook wraps _record_prediction so both should show up.
    client.post(
        "/v1/predict-single",
        files={"file": ("ok.png", _png(), "image/png")},
    )
    client.post(
        "/v1/predict-single",
        files={"file": ("bad.png", _red_png(), "image/png")},
    )

    resp = client.get("/v1/export?format=json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    ids = [r["image_ind"] for r in body["records"]]
    assert "ok.png" in ids and "bad.png" in ids
    rejected_row = next(r for r in body["records"] if r["image_ind"] == "bad.png")
    assert rejected_row["rejected"] is True
    assert rejected_row["rejection_reason"] == "not_a_marine_mammal"
    accepted_row = next(r for r in body["records"] if r["image_ind"] == "ok.png")
    assert accepted_row["rejected"] is False
    assert accepted_row["bbox"] == [0, 0, 10, 10]


def test_export_json_since_filter():
    # Insert a detection so there's something to filter
    client.post(
        "/v1/predict-single",
        files={"file": ("ok.png", _png(), "image/png")},
    )
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    resp = client.get(f"/v1/export?format=json&since={future}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["records"] == []


def test_export_json_since_invalid():
    resp = client.get("/v1/export?format=json&since=not-a-date")
    assert resp.status_code == 400
    assert "Invalid ISO-8601" in resp.json()["detail"]


# --- /v1/export CSV -------------------------------------------------------


def test_export_csv_header_only():
    resp = client.get("/v1/export?format=csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    text = resp.text
    # First line must be the header, no data rows yet
    first_line = text.splitlines()[0]
    for col in (
        "created_at",
        "image_ind",
        "class_animal",
        "id_animal",
        "probability",
        "rejected",
        "rejection_reason",
        "model_version",
        "bbox",
    ):
        assert col in first_line
    assert len(text.strip().splitlines()) == 1


def test_export_csv_with_rows():
    client.post(
        "/v1/predict-single",
        files={"file": ("ok.png", _png(), "image/png")},
    )
    client.post(
        "/v1/predict-single",
        files={"file": ("bad.png", _red_png(), "image/png")},
    )

    resp = client.get("/v1/export?format=csv")
    assert resp.status_code == 200
    reader = csv.reader(io.StringIO(resp.text))
    rows = list(reader)
    assert len(rows) == 3  # header + 2 rows
    header = rows[0]
    idx_image = header.index("image_ind")
    idx_rejected = header.index("rejected")
    idx_bbox = header.index("bbox")
    data_rows = rows[1:]
    images = {row[idx_image] for row in data_rows}
    assert images == {"ok.png", "bad.png"}
    # bbox is JSON-encoded so must round-trip through json.loads
    for row in data_rows:
        bbox = json.loads(row[idx_bbox])
        assert bbox == [0, 0, 10, 10]
        assert row[idx_rejected] in ("True", "False")


def test_export_invalid_format():
    resp = client.get("/v1/export?format=xml")
    # Query validator pattern blocks anything outside json|csv
    assert resp.status_code == 422


# --- Sanity: existing endpoints still work after routers installed -------


def test_existing_predict_single_still_works():
    resp = client.post(
        "/v1/predict-single",
        files={"file": ("sanity.png", _png(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["image_ind"] == "sanity.png"


def test_existing_predict_batch_still_works():
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        zf.writestr("a.png", _png())
    zip_buf.seek(0)
    resp = client.post(
        "/v1/predict-batch",
        files={"archive": ("b.zip", zip_buf.read(), "application/zip")},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
