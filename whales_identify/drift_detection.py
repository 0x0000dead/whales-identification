"""
Data Drift Detection Module.

Monitors distribution shifts in input data by comparing incoming image
statistics against a stored baseline. Triggers alerts when drift exceeds
a configurable threshold (default: 20%).

Usage:
    detector = DriftDetector.from_baseline("baseline_stats.json")
    result = detector.check(new_images_stats)
    if result.is_drifted:
        print(f"Drift detected: {result.drift_score:.2%}")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    is_drifted: bool
    drift_score: float
    details: dict = field(default_factory=dict)


@dataclass
class ImageStats:
    mean_rgb: list[float]
    std_rgb: list[float]
    brightness: float
    contrast: float
    sample_count: int


class DriftDetector:
    """Detects data drift by comparing image statistics to a baseline."""

    def __init__(self, baseline: ImageStats, threshold: float = 0.20):
        self.baseline = baseline
        self.threshold = threshold

    @classmethod
    def from_baseline(cls, path: str, threshold: float = 0.20):
        with open(path) as f:
            data = json.load(f)
        baseline = ImageStats(**data)
        return cls(baseline, threshold)

    @staticmethod
    def compute_stats(images: list[np.ndarray]) -> ImageStats:
        """Compute aggregate statistics from a list of RGB images."""
        means, stds, brightnesses, contrasts = [], [], [], []

        for img in images:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img_float = img.astype(np.float32) / 255.0

            means.append(img_float.mean(axis=(0, 1)).tolist())
            stds.append(img_float.std(axis=(0, 1)).tolist())

            gray = img_float.mean(axis=-1)
            brightnesses.append(float(gray.mean()))
            contrasts.append(float(gray.std()))

        return ImageStats(
            mean_rgb=np.mean(means, axis=0).tolist(),
            std_rgb=np.mean(stds, axis=0).tolist(),
            brightness=float(np.mean(brightnesses)),
            contrast=float(np.mean(contrasts)),
            sample_count=len(images),
        )

    def check(self, current: ImageStats) -> DriftResult:
        """Compare current stats to baseline. Returns drift result."""
        diffs = {}

        # Mean RGB drift
        mean_diff = np.abs(
            np.array(current.mean_rgb) - np.array(self.baseline.mean_rgb)
        ).mean()
        baseline_mean = np.array(self.baseline.mean_rgb).mean()
        diffs["mean_rgb_drift"] = float(
            mean_diff / baseline_mean if baseline_mean > 0 else 0
        )

        # Std RGB drift
        std_diff = np.abs(
            np.array(current.std_rgb) - np.array(self.baseline.std_rgb)
        ).mean()
        baseline_std = np.array(self.baseline.std_rgb).mean()
        diffs["std_rgb_drift"] = float(
            std_diff / baseline_std if baseline_std > 0 else 0
        )

        # Brightness drift
        bright_diff = abs(current.brightness - self.baseline.brightness)
        diffs["brightness_drift"] = float(
            bright_diff / self.baseline.brightness
            if self.baseline.brightness > 0
            else 0
        )

        # Contrast drift
        contrast_diff = abs(current.contrast - self.baseline.contrast)
        diffs["contrast_drift"] = float(
            contrast_diff / self.baseline.contrast if self.baseline.contrast > 0 else 0
        )

        # Overall drift score (max of individual drifts)
        drift_score = max(diffs.values())
        is_drifted = drift_score > self.threshold

        if is_drifted:
            logger.warning(
                "Data drift detected: score=%.2f (threshold=%.2f), details=%s",
                drift_score,
                self.threshold,
                diffs,
            )

        return DriftResult(
            is_drifted=is_drifted,
            drift_score=drift_score,
            details=diffs,
        )

    def save_baseline(self, stats: ImageStats, path: str):
        """Save baseline stats to JSON file."""
        data = {
            "mean_rgb": stats.mean_rgb,
            "std_rgb": stats.std_rgb,
            "brightness": stats.brightness,
            "contrast": stats.contrast,
            "sample_count": stats.sample_count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
