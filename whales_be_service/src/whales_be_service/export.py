"""Prediction history store + CSV/JSON export.

``/v1/drift-stats`` reads aggregate metrics from ``DriftMonitor``; this module
adds a parallel ring buffer that keeps the full Detection records (not just
cetacean_score aggregates) so operators can dump the last N observations as
CSV or JSON for external analysis.

The store is populated by monkey-patching ``main._record_prediction`` from
``routers.py`` — this keeps ``main.py`` untouched while still recording every
prediction that flows through the service.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections import deque
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from typing import Any

from .response_models import Detection

MAX_HISTORY = 10_000
EXPORT_FIELDS: tuple[str, ...] = (
    "created_at",
    "image_ind",
    "class_animal",
    "id_animal",
    "probability",
    "is_cetacean",
    "cetacean_score",
    "rejected",
    "rejection_reason",
    "model_version",
    "bbox",
)


class PredictionHistoryStore:
    """Thread-safe ring buffer of Detection objects with ISO-8601 timestamps.

    Behaviour:
    - Capped at ``MAX_HISTORY`` entries (oldest evicted first).
    - Every entry records the wall-clock time it was inserted so ``since``
      filtering works without relying on client clocks.
    """

    def __init__(self, max_size: int = MAX_HISTORY) -> None:
        self._buffer: deque[tuple[datetime, Detection]] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def record(self, detection: Detection) -> None:
        with self._lock:
            self._buffer.append((datetime.now(UTC), detection))

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def filter(self, since: datetime | None = None) -> list[tuple[datetime, Detection]]:
        with self._lock:
            rows = list(self._buffer)
        if since is None:
            return rows
        return [(ts, det) for ts, det in rows if ts >= since]


_default_store: PredictionHistoryStore | None = None


def get_history_store() -> PredictionHistoryStore:
    global _default_store
    if _default_store is None:
        _default_store = PredictionHistoryStore()
    return _default_store


# --- Serialisation --------------------------------------------------------


def _row_to_dict(ts: datetime, det: Detection) -> dict[str, Any]:
    data = det.model_dump()
    return {
        "created_at": ts.isoformat(),
        "image_ind": data["image_ind"],
        "class_animal": data["class_animal"],
        "id_animal": data["id_animal"],
        "probability": data["probability"],
        "is_cetacean": data["is_cetacean"],
        "cetacean_score": data["cetacean_score"],
        "rejected": data["rejected"],
        "rejection_reason": data["rejection_reason"],
        "model_version": data["model_version"],
        "bbox": data["bbox"],
    }


def rows_to_json(rows: Iterable[tuple[datetime, Detection]]) -> list[dict[str, Any]]:
    return [_row_to_dict(ts, det) for ts, det in rows]


async def stream_csv(
    rows: Iterable[tuple[datetime, Detection]],
) -> AsyncIterator[bytes]:
    """Generator for FastAPI ``StreamingResponse``.

    Yields header + one row at a time so very large exports never load the
    whole table into memory.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(EXPORT_FIELDS)
    yield buf.getvalue().encode("utf-8")
    buf.seek(0)
    buf.truncate()

    for ts, det in rows:
        row = _row_to_dict(ts, det)
        writer.writerow(
            [
                row["created_at"],
                row["image_ind"],
                row["class_animal"],
                row["id_animal"],
                row["probability"],
                row["is_cetacean"],
                row["cetacean_score"],
                row["rejected"],
                row["rejection_reason"] or "",
                row["model_version"],
                json.dumps(row["bbox"]),
            ]
        )
        yield buf.getvalue().encode("utf-8")
        buf.seek(0)
        buf.truncate()


def parse_since(value: str | None) -> datetime | None:
    """Accept ISO-8601 with or without trailing ``Z``.

    Also tolerates the common URL-encoding artefact where ``+`` in a timezone
    offset arrives as a space (browsers decode ``+`` to U+0020).
    """
    if not value:
        return None
    normalised = value.strip()
    if normalised.endswith("Z"):
        normalised = normalised[:-1] + "+00:00"
    # Fix mangled timezone offsets: "2026-04-15T00:00:00 00:00" → "...+00:00"
    if len(normalised) >= 6 and normalised[-6] == " " and normalised[-3] == ":":
        normalised = normalised[:-6] + "+" + normalised[-5:]
    try:
        dt = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
