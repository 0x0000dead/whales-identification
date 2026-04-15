"""Rolling-window drift monitor for the anti-fraud gate.

Tracks the most recent N ``cetacean_score`` values and exposes mean/std/p50
for ad-hoc inspection. If the rolling mean drops more than ``alarm_drop`` from
the calibration baseline, logs a WARNING. This is intentionally a tiny
in-memory implementation — heavyweight MLOps (MLflow, EvidentlyAI, Prometheus
remote write) plugs in on top of this when ``MLFLOW_TRACKING_URI`` is set.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from threading import Lock
from typing import Deque

logger = logging.getLogger(__name__)


class DriftMonitor:
    def __init__(
        self,
        window_size: int = 1000,
        baseline_mean: float | None = None,
        alarm_drop: float = 0.10,
    ) -> None:
        self.window_size = window_size
        self.baseline_mean = baseline_mean
        self.alarm_drop = alarm_drop

        self._scores: Deque[float] = deque(maxlen=window_size)
        self._probabilities: Deque[float] = deque(maxlen=window_size)
        self._lock = Lock()
        self._alarms_total = 0

    def record(self, cetacean_score: float, identification_probability: float) -> None:
        with self._lock:
            self._scores.append(cetacean_score)
            self._probabilities.append(identification_probability)
        self._maybe_alarm()

    def _maybe_alarm(self) -> None:
        if self.baseline_mean is None or len(self._scores) < 50:
            return
        current = statistics.fmean(self._scores)
        if (self.baseline_mean - current) >= self.alarm_drop:
            with self._lock:
                self._alarms_total += 1
            logger.warning(
                "Drift detected: rolling mean cetacean_score=%.4f, baseline=%.4f (drop>%.2f)",
                current,
                self.baseline_mean,
                self.alarm_drop,
            )

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            scores = list(self._scores)
            probs = list(self._probabilities)
        if not scores:
            return {
                "n": 0,
                "alarms_total": self._alarms_total,
                "score_mean": 0.0,
                "score_std": 0.0,
                "probability_mean": 0.0,
            }
        return {
            "n": len(scores),
            "alarms_total": self._alarms_total,
            "score_mean": round(statistics.fmean(scores), 4),
            "score_std": round(
                statistics.pstdev(scores) if len(scores) > 1 else 0.0, 4
            ),
            "probability_mean": round(statistics.fmean(probs), 4) if probs else 0.0,
        }


_default_monitor: DriftMonitor | None = None


def get_drift_monitor() -> DriftMonitor:
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = DriftMonitor()
    return _default_monitor
