"""Lightweight in-process monitoring helpers (no external dependencies)."""

from .drift import DriftMonitor, get_drift_monitor

__all__ = ["DriftMonitor", "get_drift_monitor"]
