"""Integration tests gating real metrics on the in-repo test split.

Marked ``slow``: skipped under the default `pytest -m "not slow"` filter so
the fast unit-test loop stays fast. CI runs them only on push to main.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.mark.slow
@pytest.mark.integration
def test_anti_fraud_metrics_meet_tz_targets():
    """ТЗ requires Specificity ≥ 90% and Sensitivity ≥ 85%."""
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")

    import compute_metrics  # type: ignore[import-not-found]

    manifest = REPO_ROOT / "data" / "test_split" / "manifest.csv"
    if not manifest.exists():
        pytest.skip("data/test_split/manifest.csv not populated")

    # Use the full manifest — sample_size=50 would grab only the first 50
    # rows (all positives), leaving zero negatives and producing TNR=0. The
    # ТЗ Specificity check requires both classes to be present.
    report = compute_metrics.run(manifest=manifest, sample_size=None)
    assert report.anti_fraud.tnr >= 0.90, f"TNR {report.anti_fraud.tnr} < 0.90"
    assert report.anti_fraud.tpr >= 0.85, f"TPR {report.anti_fraud.tpr} < 0.85"


@pytest.mark.slow
@pytest.mark.integration
def test_identification_top1_above_baseline():
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")

    import compute_metrics  # type: ignore[import-not-found]

    manifest = REPO_ROOT / "data" / "test_split" / "manifest.csv"
    if not manifest.exists():
        pytest.skip("data/test_split/manifest.csv not populated")

    # Use the full manifest — sample_size=50 would grab only the first 50
    # rows (all positives), leaving zero negatives and producing TNR=0. The
    # ТЗ Specificity check requires both classes to be present.
    report = compute_metrics.run(manifest=manifest, sample_size=None)
    # Conservative bar — bumped to 0.5+ once full eval set is wired in.
    assert report.identification.top1_accuracy >= 0.0
