import io
import zipfile
import base64
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from whales_be_service.main import app

client = TestClient(app)


def create_test_image_bytes(format="PNG", size=(10, 10), color=(255, 0, 0)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()


def test_predict_single_success():
    img_bytes = create_test_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    resp = client.post("/predict-single", files=files)
    assert resp.status_code == 200
    data = resp.json()
    # required keys
    for key in ("image_ind", "bbox", "class_animal", "id_animal", "probability", "mask"):
        assert key in data
    # types and basic validation
    assert data["image_ind"] == "test.png"
    assert isinstance(data["bbox"], list) and len(data["bbox"]) == 4
    assert all(isinstance(x, int) for x in data["bbox"])
    assert isinstance(data["class_animal"], str)
    assert isinstance(data["id_animal"], str)
    prob = data["probability"]
    assert isinstance(prob, float) and 0.8 <= prob <= 1.0
    mask_b64 = data["mask"]
    decoded = base64.b64decode(mask_b64)
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_predict_single_unsupported_media():
    txt_bytes = b"not an image"
    files = {"file": ("test.txt", txt_bytes, "text/plain")}
    resp = client.post("/predict-single", files=files)
    assert resp.status_code == 415
    assert "Только изображения" in resp.json()["detail"]


def test_predict_batch_success():
    # create in-memory ZIP with two images
    img1 = create_test_image_bytes()
    img2 = create_test_image_bytes(color=(0, 255, 0))
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        zf.writestr("img1.png", img1)
        zf.writestr("subdir/img2.jpg", img2)
    zip_buf.seek(0)

    files = {"archive": ("batch.zip", zip_buf.read(), "application/zip")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 200
    results = resp.json()
    assert isinstance(results, list) and len(results) == 2

    for det in results:
        for key in ("image_ind", "bbox", "class_animal", "id_animal", "probability", "mask"):
            assert key in det
        assert isinstance(det["image_ind"], str)
        assert isinstance(det["bbox"], list) and len(det["bbox"]) == 4
        assert all(isinstance(x, int) for x in det["bbox"])
        prob = det["probability"]
        assert isinstance(prob, float) and 0.8 <= prob <= 1.0
        decoded = base64.b64decode(det["mask"])
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_predict_batch_wrong_content_type():
    img_bytes = create_test_image_bytes()
    files = {"archive": ("notazip.png", img_bytes, "image/png")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 415
    assert "Ожидается ZIP-архив" in resp.json()["detail"]


def test_predict_batch_bad_zip():
    bad = b"this is not a zip"
    files = {"archive": ("bad.zip", bad, "application/zip")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 400
    assert "Не удаётся распаковать архив" in resp.json()["detail"]
