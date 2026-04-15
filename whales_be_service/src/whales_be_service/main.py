"""FastAPI entry point for the EcoMarineAI HTTP service.

This module wires the routes; all ML logic is delegated to
``inference.InferencePipeline``, which is built once at startup via the
FastAPI lifespan context.
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator
from zipfile import BadZipFile, ZipFile

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.routing import APIRouter
from PIL import Image, UnidentifiedImageError
from starlette.middleware.cors import CORSMiddleware

from .inference import get_pipeline
from .inference.pipeline import InferencePipeline
from .monitoring import get_drift_monitor
from .response_models import Detection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pipeline = get_pipeline()
    pipeline.warmup()
    app.state.pipeline = pipeline
    logger.info("EcoMarineAI service ready.")
    yield


app = FastAPI(
    title="EcoMarineAI Identification API",
    version="1.1.0",
    description="Идентификация морских млекопитающих по аэрофотоснимкам с CLIP-антифрод гейтом.",
    lifespan=lifespan,
)


_default_origins = "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173,http://127.0.0.1:8080"
_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
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

    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Превышен лимит запросов. Повторите позже."},
        )

    _rate_limit_store[client_ip].append(now)
    return await call_next(request)


# --- Metrics collection ---
_metrics: dict[str, object] = {
    "start_time": time.time(),
    "requests_total": 0,
    "requests_by_endpoint": defaultdict(int),
    "errors_total": 0,
    "latency_sum_ms": 0.0,
    "latency_count": 0,
    "predictions_total": 0,
    "rejections_total": 0,
    "rejections_by_reason": defaultdict(int),
    "cetacean_score_sum": 0.0,
    "cetacean_score_count": 0,
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


def _record_prediction(det: Detection) -> None:
    if det.rejected:
        _metrics["rejections_total"] += 1
        if det.rejection_reason:
            _metrics["rejections_by_reason"][det.rejection_reason] += 1
    else:
        _metrics["predictions_total"] += 1
    _metrics["cetacean_score_sum"] += det.cetacean_score
    _metrics["cetacean_score_count"] += 1
    get_drift_monitor().record(det.cetacean_score, det.probability)


def get_pipeline_dep(request: Request) -> InferencePipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        pipeline = get_pipeline()
        request.app.state.pipeline = pipeline
    return pipeline


# --- Health & Metrics endpoints (root level, no versioning) ---


@app.get("/health", summary="Health check endpoint")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", summary="Prometheus-compatible metrics")
async def metrics() -> PlainTextResponse:
    avg_latency = (
        _metrics["latency_sum_ms"] / _metrics["latency_count"]
        if _metrics["latency_count"] > 0
        else 0
    )
    avg_cetacean_score = (
        _metrics["cetacean_score_sum"] / _metrics["cetacean_score_count"]
        if _metrics["cetacean_score_count"] > 0
        else 0
    )
    uptime = time.time() - _metrics["start_time"]
    total_req = _metrics["requests_total"]
    avail = (
        (total_req - _metrics["errors_total"]) / total_req * 100
        if total_req > 0
        else 100.0
    )
    lines = [
        "# HELP uptime_seconds Seconds since the service started",
        "# TYPE uptime_seconds counter",
        f"uptime_seconds {uptime:.1f}",
        "# HELP availability_percent Service availability (% of non-5xx requests)",
        "# TYPE availability_percent gauge",
        f"availability_percent {avail:.3f}",
        "# HELP requests_total Total HTTP requests",
        "# TYPE requests_total counter",
        f'requests_total {_metrics["requests_total"]}',
        "# HELP errors_total Total HTTP errors (4xx/5xx)",
        "# TYPE errors_total counter",
        f'errors_total {_metrics["errors_total"]}',
        "# HELP predictions_total Successful (not rejected) predictions",
        "# TYPE predictions_total counter",
        f'predictions_total {_metrics["predictions_total"]}',
        "# HELP rejections_total Anti-fraud / low-confidence rejections",
        "# TYPE rejections_total counter",
        f'rejections_total {_metrics["rejections_total"]}',
        "# HELP latency_avg_ms Average request latency in ms",
        "# TYPE latency_avg_ms gauge",
        f"latency_avg_ms {avg_latency:.2f}",
        "# HELP cetacean_score_avg Rolling mean CLIP cetacean score",
        "# TYPE cetacean_score_avg gauge",
        f"cetacean_score_avg {avg_cetacean_score:.4f}",
    ]
    for reason, count in _metrics["rejections_by_reason"].items():
        lines.append(f'rejections_by_reason{{reason="{reason}"}} {count}')
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


DETECTION_EXAMPLE = {
    "image_ind": "whale_photo.jpg",
    "bbox": [0, 0, 1920, 1080],
    "class_animal": "1a71fbb72250",
    "id_animal": "humpback_whale",
    "probability": 0.934,
    "mask": "iVBORw0KGgo...",
    "is_cetacean": True,
    "cetacean_score": 0.87,
    "rejected": False,
    "rejection_reason": None,
    "model_version": "vit_l32-v1",
}

REJECTION_EXAMPLE = {
    "image_ind": "screenshot.png",
    "bbox": [0, 0, 800, 600],
    "class_animal": "",
    "id_animal": "unknown",
    "probability": 0.0,
    "mask": None,
    "is_cetacean": False,
    "cetacean_score": 0.12,
    "rejected": True,
    "rejection_reason": "not_a_marine_mammal",
    "model_version": "vit_l32-v1",
}


# --- API v1 Router ---
v1 = APIRouter(prefix="/v1", tags=["v1"])


@v1.post(
    "/predict-single",
    response_model=Detection,
    summary="Фото → JSON с результатом идентификации (или rejection)",
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "accepted": {"value": DETECTION_EXAMPLE},
                        "rejected": {"value": REJECTION_EXAMPLE},
                    }
                }
            }
        },
        415: {
            "description": "Unsupported media type",
            "content": {
                "application/json": {"example": {"detail": "Только изображения."}}
            },
        },
    },
)
async def predict_single_v1(
    file: UploadFile = File(...),
    pipeline: InferencePipeline = Depends(get_pipeline_dep),
) -> Detection:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Только изображения.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Пустой файл.")

    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(415, "Не удалось распознать изображение.") from None

    detection = pipeline.predict(
        pil_img=pil_img,
        filename=file.filename or "unknown",
        img_bytes=data,
        generate_mask=True,
    )
    _record_prediction(detection)
    return detection


@v1.post(
    "/predict-batch",
    summary="ZIP с изображениями → JSON-массив результатов",
    responses={
        200: {
            "content": {"application/json": {"example": [DETECTION_EXAMPLE, REJECTION_EXAMPLE]}}
        },
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
async def predict_batch_v1(
    archive: UploadFile = File(...),
    pipeline: InferencePipeline = Depends(get_pipeline_dep),
) -> JSONResponse:
    if archive.content_type not in (
        "application/zip",
        "application/x-zip-compressed",
    ):
        raise HTTPException(415, "Ожидается ZIP-архив.")

    raw = await archive.read()
    try:
        zf = ZipFile(io.BytesIO(raw))
    except BadZipFile as e:
        raise HTTPException(400, "Не удаётся распаковать архив.") from e

    results: list[dict] = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        try:
            img_bytes = zf.read(name)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except (KeyError, UnidentifiedImageError):
            continue
        except Exception:  # nosec B112 - skip corrupted files
            continue

        detection = pipeline.predict(
            pil_img=pil_img,
            filename=name,
            img_bytes=img_bytes,
            generate_mask=False,  # skip slow rembg in batch mode
        )
        _record_prediction(detection)
        results.append(detection.model_dump())

    zf.close()
    return JSONResponse(content=results)


@v1.get("/drift-stats", summary="Rolling-window cetacean_score statistics")
async def drift_stats() -> dict:
    return get_drift_monitor().stats()


app.include_router(v1)


# --- Backward-compatible root endpoints (delegate to v1) ---


@app.post(
    "/predict-single",
    response_model=Detection,
    summary="Legacy alias → /v1/predict-single",
)
async def predict_single(
    file: UploadFile = File(...),
    pipeline: InferencePipeline = Depends(get_pipeline_dep),
) -> Detection:
    return await predict_single_v1(file=file, pipeline=pipeline)


@app.post(
    "/predict-batch",
    summary="Legacy alias → /v1/predict-batch",
)
async def predict_batch(
    archive: UploadFile = File(...),
    pipeline: InferencePipeline = Depends(get_pipeline_dep),
) -> JSONResponse:
    return await predict_batch_v1(archive=archive, pipeline=pipeline)
