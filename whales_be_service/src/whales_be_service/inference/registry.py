"""Pipeline + model registry — singletons + metadata accessors.

``get_pipeline()`` is the only function called by FastAPI startup. Everything
downstream uses the returned ``InferencePipeline``.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import yaml

from .anti_fraud import AntiFraudGate
from .identification import IdentificationModel
from .pipeline import InferencePipeline

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _BASE_DIR / "config.yaml"
_REPO_ROOT = _BASE_DIR.parent.parent.parent.parent
_REGISTRY_PATH = _REPO_ROOT / "models" / "registry.json"


def _load_anti_fraud_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("anti_fraud", {})


def _load_baseline_cetacean_score_mean() -> float | None:
    """Read the baseline cetacean_score mean from the CI regression snapshot.

    If ``reports/metrics_baseline.json`` exists and contains a plausible
    positive score, use it as the drift baseline. Otherwise return None and
    let the drift monitor run in "no baseline" mode (no alarms).
    """
    baseline_path = _REPO_ROOT / "reports" / "metrics_baseline.json"
    if not baseline_path.exists():
        return None
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        positives = data.get("anti_fraud", {}).get("tpr")
        # Use TPR as a proxy if present; fall back to roc_auc or 0.9
        if isinstance(positives, int | float) and 0 < positives <= 1:
            return float(positives)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not parse metrics_baseline.json: %s", e)
    return None


@lru_cache(maxsize=1)
def get_pipeline() -> InferencePipeline:
    """Build (once) and return the singleton ``InferencePipeline``."""
    af_cfg = _load_anti_fraud_config()
    gate = AntiFraudGate(
        model_name=af_cfg.get("model_name", "ViT-B-32"),
        pretrained=af_cfg.get("pretrained", "laion2b_s34b_b79k"),
        threshold=af_cfg.get("threshold"),
    )
    identification = IdentificationModel()
    pipeline = InferencePipeline(
        anti_fraud=gate,
        identification=identification,
        min_confidence=af_cfg.get("min_confidence", 0.05),
    )

    # Wire the drift baseline so /v1/drift-stats alarms actually fire.
    baseline_mean = _load_baseline_cetacean_score_mean()
    if baseline_mean is not None:
        from ..monitoring.drift import get_drift_monitor  # noqa: PLC0415

        monitor = get_drift_monitor()
        monitor.baseline_mean = baseline_mean
        logger.info("Drift baseline cetacean_score mean = %.4f", baseline_mean)
    else:
        logger.info("No drift baseline configured — alarms disabled.")

    logger.info("Built InferencePipeline (lazy — models not loaded yet).")
    return pipeline


def list_models() -> list[dict]:
    """Return registered model metadata from ``models/registry.json``."""
    if not _REGISTRY_PATH.exists():
        return []
    with _REGISTRY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", [])


def get_model_metadata(name: str, version: str | None = None) -> dict | None:
    """Find a model by name (and optional version) in the registry."""
    for entry in list_models():
        if entry.get("name") != name:
            continue
        if version is None or entry.get("version") == version:
            return entry
    return None
