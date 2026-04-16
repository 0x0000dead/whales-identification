"""Pipeline + model registry — singletons + metadata accessors.

``get_pipeline()`` is the only function called by FastAPI startup. Everything
downstream uses the returned ``InferencePipeline`` (or ``EnsemblePipeline``
when ``active_model: ensemble`` is set in ``models_config.yaml``).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import yaml

from .anti_fraud import AntiFraudGate
from .ensemble import EnsemblePipeline, build_ensemble_from_config
from .identification import IdentificationModel
from .pipeline import InferencePipeline

logger = logging.getLogger(__name__)

PipelineLike = InferencePipeline | EnsemblePipeline

_BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _BASE_DIR / "config.yaml"
_REPO_ROOT = _BASE_DIR.parent.parent.parent.parent
_REGISTRY_PATH = _REPO_ROOT / "models" / "registry.json"
_MODELS_CONFIG_PATH = _REPO_ROOT / "models_config.yaml"


def _load_anti_fraud_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("anti_fraud", {})


def _load_models_config() -> dict:
    """Parse ``models_config.yaml`` — returns empty dict if missing so default
    single-model pipeline remains the fallback."""
    if not _MODELS_CONFIG_PATH.exists():
        return {}
    try:
        with _MODELS_CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Could not parse models_config.yaml: %s", e)
        return {}


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
def get_pipeline() -> PipelineLike:
    """Build (once) and return the singleton pipeline.

    Returns an ``EnsemblePipeline`` when ``models_config.yaml`` sets
    ``active_model: ensemble`` — otherwise the default single-model
    ``InferencePipeline``. The caller does not need to know which one it
    received: both expose the same ``predict()`` / ``warmup()`` /
    ``model_version`` contract.
    """
    af_cfg = _load_anti_fraud_config()
    gate = AntiFraudGate(
        model_name=af_cfg.get("model_name", "ViT-B-32"),
        pretrained=af_cfg.get("pretrained", "laion2b_s34b_b79k"),
        threshold=af_cfg.get("threshold"),
    )
    identification = IdentificationModel()

    models_cfg = _load_models_config()
    active_model = str(models_cfg.get("active_model", "effb4_arcface"))

    # Refuse to activate anything flagged `deprecated: true` — the record is
    # kept only for reproducibility of Stage 1 experiments, the weights file
    # is not auto-downloaded, and falling through would crash at first
    # `torch.load()` with a confusing FileNotFoundError. Fail fast with a
    # clear error instead.
    model_block = (models_cfg.get("models") or {}).get(active_model) or {}
    if model_block.get("deprecated", False):
        reason = model_block.get(
            "deprecated_reason", "marked deprecated in models_config.yaml"
        )
        raise RuntimeError(
            f"active_model='{active_model}' is deprecated and cannot be loaded: {reason}. "
            f"Set active_model to 'effb4_arcface' (production) or 'ensemble'."
        )

    pipeline: PipelineLike
    if active_model == "ensemble":
        ensemble_block = (models_cfg.get("models") or {}).get("ensemble") or {}
        ensemble_config = build_ensemble_from_config(ensemble_block)
        pipeline = EnsemblePipeline(
            anti_fraud=gate,
            identification=identification,
            config=ensemble_config,
        )
        logger.info(
            "Built EnsemblePipeline with stages=%s (active_model=ensemble).",
            ensemble_config.active_stages,
        )
    else:
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
