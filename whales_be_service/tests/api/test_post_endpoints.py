import io
import zipfile

from fastapi.testclient import TestClient
from PIL import Image

from whales_be_service.main import app

client = TestClient(app)


def create_test_image_bytes(format="PNG", size=(10, 10), color=(0, 200, 0)):
    """Default colour is GREEN — the StubPipeline accepts everything that is
    NOT red. Red images are routed to the rejection branch.
    """
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()


def _create_red_image_bytes():
    return create_test_image_bytes(color=(255, 0, 0))


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_single_accepted():
    files = {"file": ("whale.png", create_test_image_bytes(), "image/png")}
    resp = client.post("/predict-single", files=files)
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "image_ind",
        "bbox",
        "class_animal",
        "id_animal",
        "probability",
        "is_cetacean",
        "cetacean_score",
        "rejected",
        "rejection_reason",
        "model_version",
    ):
        assert key in data
    assert data["image_ind"] == "whale.png"
    assert data["is_cetacean"] is True
    assert data["rejected"] is False
    assert data["rejection_reason"] is None
    assert data["probability"] > 0
    assert isinstance(data["bbox"], list) and len(data["bbox"]) == 4


def test_predict_single_rejected_returns_200():
    """Anti-fraud rejection is a successful classification, not an HTTP error."""
    files = {"file": ("text_screenshot.png", _create_red_image_bytes(), "image/png")}
    resp = client.post("/predict-single", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_cetacean"] is False
    assert data["rejected"] is True
    assert data["rejection_reason"] == "not_a_marine_mammal"
    assert data["probability"] == 0.0


def test_predict_single_unsupported_media():
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    resp = client.post("/predict-single", files=files)
    assert resp.status_code == 415
    assert "Только изображения" in resp.json()["detail"]


def test_predict_single_empty_file():
    files = {"file": ("empty.png", b"", "image/png")}
    resp = client.post("/predict-single", files=files)
    assert resp.status_code == 400
    assert "Пустой файл" in resp.json()["detail"]


def test_predict_batch_success():
    accepted = create_test_image_bytes()
    rejected = _create_red_image_bytes()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        zf.writestr("whale1.png", accepted)
        zf.writestr("subdir/whale2.png", accepted)
        zf.writestr("noise.png", rejected)
    zip_buf.seek(0)

    files = {"archive": ("batch.zip", zip_buf.read(), "application/zip")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 3

    accepted_names = [r["image_ind"] for r in results if not r["rejected"]]
    rejected_names = [r["image_ind"] for r in results if r["rejected"]]
    assert sorted(accepted_names) == ["subdir/whale2.png", "whale1.png"]
    assert rejected_names == ["noise.png"]
    for r in results:
        if r["rejected"]:
            assert r["rejection_reason"] == "not_a_marine_mammal"


def test_predict_batch_wrong_content_type():
    files = {"archive": ("notazip.png", create_test_image_bytes(), "image/png")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 415


def test_predict_batch_bad_zip():
    files = {"archive": ("bad.zip", b"this is not a zip", "application/zip")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 400


def test_predict_batch_empty_zip():
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w"):
        pass
    zip_buf.seek(0)
    files = {"archive": ("empty.zip", zip_buf.read(), "application/zip")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 200
    assert resp.json() == []


def test_v1_predict_single():
    files = {"file": ("v1_test.png", create_test_image_bytes(), "image/png")}
    resp = client.post("/v1/predict-single", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["image_ind"] == "v1_test.png"
    assert data["is_cetacean"] is True


def test_v1_predict_batch():
    img = create_test_image_bytes()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        zf.writestr("a.png", img)
    zip_buf.seek(0)
    files = {"archive": ("v1.zip", zip_buf.read(), "application/zip")}
    resp = client.post("/v1/predict-batch", files=files)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "requests_total" in body
    assert "errors_total" in body
    assert "predictions_total" in body
    assert "rejections_total" in body
    assert "cetacean_score_avg" in body
