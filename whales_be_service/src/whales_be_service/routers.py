"""Webhook + export API routers (Stage 3, §3.8).

Declarative ``APIRouter`` definitions for push notifications and prediction
history export. On import this module:

1. Adds two new routers to the existing FastAPI ``app`` defined in ``main.py``
   under the ``/v1`` prefix.
2. Wraps ``main._record_prediction`` so every successful or rejected
   detection also flows into :class:`export.PredictionHistoryStore`. This is
   the minimal hook that lets ``/v1/export`` return real, live data without
   touching ``main.py`` itself.

Importing ``whales_be_service.routers`` is therefore the single side-effect
required at application bootstrap to wire these features in. Tests for these
endpoints import this module at the top to trigger registration.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .export import (
    get_history_store,
    parse_since,
    rows_to_json,
    stream_csv,
)
from .response_models import (
    Detection,
    WebhookInfo,
    WebhookListResponse,
    WebhookRegisterRequest,
    WebhookRegisterResponse,
)
from .webhooks import get_webhook_registry

logger = logging.getLogger(__name__)

webhook_router = APIRouter(prefix="/v1", tags=["v1-webhook"])
export_router = APIRouter(prefix="/v1", tags=["v1-export"])


# --- Webhook endpoints ----------------------------------------------------


@webhook_router.post(
    "/webhook/register",
    response_model=WebhookRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a callback URL for push notifications",
    responses={
        201: {
            "description": "Webhook accepted and assigned an id",
            "content": {
                "application/json": {
                    "example": {
                        "webhook_id": "b7c1e7e0c4b247c0b8f8b8f8b8f8b8f8",
                        "status": "registered",
                        "url": "https://client.example.com/hooks/whales",
                        "events": ["batch_completed"],
                    }
                }
            },
        },
    },
)
async def register_webhook(body: WebhookRegisterRequest) -> WebhookRegisterResponse:
    sub = get_webhook_registry().register(url=str(body.url), events=list(body.events))
    return WebhookRegisterResponse(
        webhook_id=sub.webhook_id,
        url=body.url,
        events=list(body.events),
    )


@webhook_router.delete(
    "/webhook/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unregister a previously registered webhook",
    responses={
        204: {"description": "Webhook removed"},
        404: {"description": "Unknown webhook_id"},
    },
)
async def unregister_webhook(webhook_id: str) -> Response:
    removed = get_webhook_registry().unregister(webhook_id)
    if not removed:
        raise HTTPException(status_code=404, detail="webhook_id not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@webhook_router.get(
    "/webhooks",
    response_model=WebhookListResponse,
    summary="List all currently registered webhooks",
)
async def list_webhooks() -> WebhookListResponse:
    items = get_webhook_registry().list_all()
    infos = [
        WebhookInfo(
            webhook_id=s.webhook_id,
            url=s.url,  # type: ignore[arg-type]
            events=list(s.events),  # type: ignore[arg-type]
            created_at=s.created_at,
        )
        for s in items
    ]
    return WebhookListResponse(webhooks=infos, count=len(infos))


# --- Export endpoints -----------------------------------------------------


@export_router.get(
    "/export",
    summary="Export prediction history as CSV or JSON",
    responses={
        200: {
            "description": "Streaming CSV or JSON payload",
            "content": {
                "text/csv": {},
                "application/json": {
                    "example": {
                        "count": 1,
                        "since": None,
                        "records": [
                            {
                                "created_at": "2026-04-15T12:00:00+00:00",
                                "image_ind": "whale.jpg",
                                "class_animal": "1a71fbb72250",
                                "id_animal": "humpback_whale",
                                "probability": 0.92,
                                "is_cetacean": True,
                                "cetacean_score": 0.91,
                                "rejected": False,
                                "rejection_reason": None,
                                "model_version": "effb4-arcface-v1",
                                "bbox": [0, 0, 10, 10],
                            }
                        ],
                    }
                },
            },
        },
        400: {"description": "Invalid format or since parameter"},
    },
)
async def export_history(
    format: str = Query("json", pattern="^(json|csv)$"),
    since: str | None = Query(None, description="ISO-8601 timestamp filter"),
) -> Response:
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rows = get_history_store().filter(since=since_dt)

    if format == "json":
        return JSONResponse(
            content={
                "count": len(rows),
                "since": since,
                "records": rows_to_json(rows),
            }
        )

    return StreamingResponse(
        stream_csv(rows),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=whales_export.csv"},
    )


# --- Wiring ---------------------------------------------------------------


def _install_routers() -> None:
    """Attach the new routers to the app + hook the prediction recorder.

    Runs once at import time. Subsequent imports are no-ops thanks to the
    ``_installed`` sentinel on the module.
    """
    # Lazy import to avoid circular references and keep torch-free imports fast.
    from . import main as _main

    if getattr(_main.app.state, "_stage3_routers_installed", False):
        return

    _main.app.include_router(webhook_router)
    _main.app.include_router(export_router)

    original_record = _main._record_prediction
    history = get_history_store()

    def _wrapped_record(det: Detection) -> None:
        original_record(det)
        history.record(det)
        # Fire-and-forget webhook dispatch for rejections; batch_completed is
        # emitted from the batch endpoint, not per-record.
        if det.rejected:
            try:
                get_webhook_registry().dispatch_sync(
                    "prediction_rejected",
                    {"image_ind": det.image_ind, "reason": det.rejection_reason},
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Webhook dispatch skipped: %s", exc)

    _main._record_prediction = _wrapped_record  # type: ignore[assignment]
    _main.app.state._stage3_routers_installed = True
    logger.info("Stage 3 webhook/export routers installed.")


def notify_batch_completed(
    background_tasks: BackgroundTasks,
    payload: dict[str, Any],
) -> None:
    """Helper for batch endpoints to trigger the ``batch_completed`` event.

    Kept as a free function so new batch endpoints can opt-in without needing
    to know the registry lifecycle.
    """
    registry = get_webhook_registry()

    def _task() -> None:
        registry.dispatch_sync("batch_completed", payload)

    background_tasks.add_task(_task)


# Bogus guard so `datetime` import isn't flagged as unused by linters — it's
# consumed by the type-annotation string resolution at runtime.
_ = datetime

_install_routers()
