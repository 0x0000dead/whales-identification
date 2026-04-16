"""In-memory webhook registry + async dispatcher.

Used by `routers.py` to expose `/v1/webhook/*` endpoints for push notifications
about batch prediction completion. Intentionally minimal: thread-safe dict,
UUID-based ids, short TTL on in-flight deliveries.

TODO: persist в Redis для production (see ``WebhookRegistry`` docstring).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Event names that can be subscribed to. Keep this closed so registration
# validates the ``events`` field at request time.
SUPPORTED_EVENTS: frozenset[str] = frozenset({"batch_completed", "prediction_rejected"})


@dataclass
class WebhookSubscription:
    webhook_id: str
    url: str
    events: list[str]
    created_at: float = field(default_factory=lambda: __import__("time").time())


class WebhookRegistry:
    """Thread-safe in-memory registry.

    Backs `/v1/webhook/register`, `/v1/webhook/{id}` and `/v1/webhooks`. A real
    production deployment should persist this in Redis or Postgres; for the
    Stage 3 ФСИ demo the in-memory implementation is sufficient and explicit.
    """

    def __init__(self) -> None:
        self._store: dict[str, WebhookSubscription] = {}
        self._lock = threading.Lock()

    # --- CRUD -----------------------------------------------------------------

    def register(self, url: str, events: list[str]) -> WebhookSubscription:
        webhook_id = uuid.uuid4().hex
        sub = WebhookSubscription(webhook_id=webhook_id, url=url, events=list(events))
        with self._lock:
            self._store[webhook_id] = sub
        logger.info("Webhook registered: %s → %s (events=%s)", webhook_id, url, events)
        return sub

    def unregister(self, webhook_id: str) -> bool:
        with self._lock:
            return self._store.pop(webhook_id, None) is not None

    def list_all(self) -> list[WebhookSubscription]:
        with self._lock:
            return list(self._store.values())

    def get(self, webhook_id: str) -> WebhookSubscription | None:
        with self._lock:
            return self._store.get(webhook_id)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def subscribers_for(self, event: str) -> list[WebhookSubscription]:
        with self._lock:
            return [s for s in self._store.values() if event in s.events]

    # --- Dispatch -------------------------------------------------------------

    async def dispatch(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 5.0,
    ) -> list[dict[str, Any]]:
        """POST the payload to every subscriber of the given event.

        Returns per-subscription delivery results. Exceptions never propagate —
        webhook delivery is best-effort. Pass a pre-built ``client`` from tests
        to substitute ``MockTransport``.
        """
        subs = self.subscribers_for(event)
        if not subs:
            return []

        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(timeout=timeout)

        results: list[dict[str, Any]] = []
        try:
            for sub in subs:
                body = {"event": event, "webhook_id": sub.webhook_id, "data": payload}
                try:
                    assert client is not None  # narrow for mypy
                    resp = await client.post(sub.url, json=body)
                    results.append(
                        {
                            "webhook_id": sub.webhook_id,
                            "status_code": resp.status_code,
                            "ok": resp.is_success,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort delivery
                    logger.warning(
                        "Webhook delivery failed: %s → %s (%s)",
                        sub.webhook_id,
                        sub.url,
                        exc,
                    )
                    results.append(
                        {
                            "webhook_id": sub.webhook_id,
                            "status_code": None,
                            "ok": False,
                            "error": str(exc),
                        }
                    )
        finally:
            if own_client and client is not None:
                await client.aclose()
        return results

    def dispatch_sync(self, event: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget wrapper for FastAPI ``BackgroundTasks``.

        BackgroundTasks accepts a callable; this wraps the async dispatcher in
        a short-lived event loop so the request thread is never blocked. When
        no subscribers exist we skip the loop entirely — this keeps the hot
        path in ``_record_prediction`` free from asyncio overhead.
        """
        if not self.subscribers_for(event):
            return
        try:
            try:
                asyncio.get_running_loop()
                # An outer loop is running (e.g. TestClient). Spin up a
                # worker thread so we can safely ``asyncio.run`` inside it.
                import threading

                done = threading.Event()

                def _runner() -> None:
                    try:
                        asyncio.run(self.dispatch(event, payload))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Webhook worker failed: %s", exc)
                    finally:
                        done.set()

                threading.Thread(target=_runner, daemon=True).start()
                done.wait(timeout=10.0)
            except RuntimeError:
                asyncio.run(self.dispatch(event, payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Background webhook dispatch failed: %s", exc)


_default_registry: WebhookRegistry | None = None


def get_webhook_registry() -> WebhookRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = WebhookRegistry()
    return _default_registry
