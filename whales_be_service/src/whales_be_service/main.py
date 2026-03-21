import base64
import io
import random
import time
from collections import defaultdict
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.routing import APIRouter
from PIL import Image, UnidentifiedImageError
from starlette.middleware.cors import CORSMiddleware

from .response_models import Detection, generate_base64_mask_with_removed_background

app = FastAPI(
    title="Whales Identification API",
    version="1.0.0",
    description="API для идентификации морских млекопитающих по аэрофотоснимкам",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Rate limiting (in-memory, per-IP) ---
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60  # seconds
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Превышен лимит запросов. Повторите позже."},
        )

    _rate_limit_store[client_ip].append(now)
    response = await call_next(request)
    return response


# --- Metrics collection ---
_metrics = {
    "requests_total": 0,
    "requests_by_endpoint": defaultdict(int),
    "errors_total": 0,
    "latency_sum_ms": 0.0,
    "latency_count": 0,
    "predictions_total": 0,
}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    _metrics["requests_total"] += 1
    _metrics["requests_by_endpoint"][request.url.path] += 1
    _metrics["latency_sum_ms"] += duration_ms
    _metrics["latency_count"] += 1

    if response.status_code >= 400:
        _metrics["errors_total"] += 1

    return response


# --- Health & Metrics endpoints (root level, no versioning) ---


@app.get("/health", summary="Health check endpoint")
async def health():
    return {"status": "ok"}


@app.get("/metrics", summary="Prometheus-compatible metrics")
async def metrics():
    avg_latency = (
        _metrics["latency_sum_ms"] / _metrics["latency_count"]
        if _metrics["latency_count"] > 0
        else 0
    )
    lines = [
        "# HELP requests_total Total HTTP requests",
        "# TYPE requests_total counter",
        f'requests_total {_metrics["requests_total"]}',
        "# HELP errors_total Total HTTP errors (4xx/5xx)",
        "# TYPE errors_total counter",
        f'errors_total {_metrics["errors_total"]}',
        "# HELP predictions_total Total predictions made",
        "# TYPE predictions_total counter",
        f'predictions_total {_metrics["predictions_total"]}',
        "# HELP latency_avg_ms Average request latency in ms",
        "# TYPE latency_avg_ms gauge",
        f"latency_avg_ms {avg_latency:.2f}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


# --- Load config ---
BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
ID_TO_NAME = cfg.get("id_to_name", {})


DETECTION_EXAMPLE = {
    "image_ind": "whale_photo.jpg",
    "bbox": [10, 20, 300, 200],
    "class_animal": "cadddb1636b9",
    "id_animal": "humpback_whale",
    "probability": 0.934,
    "mask": "iVBORw0KGgo...",
}


def detection_id(filename: str, img_bytes: bytes) -> dict:
    bbox = [random.randint(0, 50) for _ in range(4)]  # nosec B311
    class_id = "cadddb1636b9"
    prob = round(random.uniform(0.8, 1.0), 3)  # nosec B311 - mock data

    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    mask_b64 = generate_base64_mask_with_removed_background(img_bytes)

    _metrics["predictions_total"] += 1

    return {
        "image_ind": filename,
        "bbox": bbox,
        "class_animal": class_id,
        "id_animal": ID_TO_NAME.get(class_id, class_id),
        "probability": prob,
        "mask": mask_b64,
    }


# --- API v1 Router ---
v1 = APIRouter(prefix="/v1", tags=["v1"])


@v1.post(
    "/predict-single",
    response_model=Detection,
    summary="Фото → JSON с bbox+mask",
    responses={
        200: {"content": {"application/json": {"example": DETECTION_EXAMPLE}}},
        415: {
            "description": "Unsupported media type",
            "content": {
                "application/json": {"example": {"detail": "Только изображения."}}
            },
        },
    },
)
async def predict_single_v1(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Только изображения.")
    data = await file.read()
    det = detection_id(file.filename, data)
    return JSONResponse(content=det)


@v1.post(
    "/predict-batch",
    summary="ZIP → JSON[]",
    responses={
        200: {"content": {"application/json": {"example": [DETECTION_EXAMPLE]}}},
        400: {
            "description": "Invalid ZIP archive",
            "content": {
                "application/json": {
                    "example": {"detail": "Не удаётся распаковать архив."}
                }
            },
        },
        415: {
            "description": "Unsupported media type",
            "content": {
                "application/json": {"example": {"detail": "Ожидается ZIP-архив."}}
            },
        },
    },
)
async def predict_batch_v1(archive: UploadFile = File(...)):
    if archive.content_type not in (
        "application/zip",
        "application/x-zip-compressed",
    ):
        raise HTTPException(415, "Ожидается ZIP-архив.")

    raw = await archive.read()
    try:
        zf = ZipFile(io.BytesIO(raw))
    except BadZipFile:
        raise HTTPException(400, "Не удаётся распаковать архив.")

    results: list[dict] = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        try:
            img_bytes = zf.read(name)
            with Image.open(io.BytesIO(img_bytes)) as img:
                img.verify()
            det = detection_id(name, img_bytes)
            results.append(det)
        except (KeyError, UnidentifiedImageError):
            continue
        except Exception:  # nosec B112 - skip corrupted files
            continue

    zf.close()
    return JSONResponse(content=results)


app.include_router(v1)


# --- Backward-compatible root endpoints (delegate to v1) ---


@app.post(
    "/predict-single",
    response_model=Detection,
    summary="Фото → JSON с bbox+mask",
    responses={
        200: {"content": {"application/json": {"example": DETECTION_EXAMPLE}}},
    },
)
async def predict_single(file: UploadFile = File(...)):
    return await predict_single_v1(file)


@app.post(
    "/predict-batch",
    summary="ZIP → JSON[]",
)
async def predict_batch(archive: UploadFile = File(...)):
    return await predict_batch_v1(archive)
