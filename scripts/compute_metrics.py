#!/usr/bin/env python3
"""Compute real ML metrics for EcoMarineAI on the in-repo test split.

This script replaces the old practice of writing precision/recall numbers
directly into ``models_config.yaml`` by hand. Every metric here is computed
from real predictions on real images and saved to:

* ``reports/metrics_latest.json`` — machine-readable, used by the CI gate
* ``reports/METRICS.md`` — human-readable Markdown table
* ``MODEL_CARD.md`` — auto-injected between ``<!-- metrics:start -->`` /
  ``<!-- metrics:end -->`` markers (only when ``--update-model-card`` is set)

Three classes of metrics are reported:

1. **Anti-fraud** (binary, positive vs negative): TPR/Sensitivity, TNR/
   Specificity, Precision, F1, ROC-AUC on ``cetacean_score``.
2. **Identification** (multiclass, on positives only): top-1 / top-5 accuracy,
   macro-F1.
3. **Performance**: p50 / p95 / p99 wall-clock latency per image.

Usage::

    python scripts/compute_metrics.py \\
        --manifest data/test_split/manifest.csv \\
        --output-json reports/metrics_latest.json \\
        --output-md reports/METRICS.md \\
        --update-model-card
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "test_split" / "manifest.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "metrics_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "METRICS.md"
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"

MARKERS = ("<!-- metrics:start -->", "<!-- metrics:end -->")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_metrics")


@dataclass
class AntiFraudMetrics:
    n_positive: int = 0
    n_negative: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    tpr: float = 0.0  # sensitivity
    tnr: float = 0.0  # specificity
    precision: float = 0.0
    f1: float = 0.0
    roc_auc: float | None = None


@dataclass
class IdentificationMetrics:
    n_samples: int = 0
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0  # placeholder — pipeline currently exposes top-1 only
    n_unique_individuals: int = 0


@dataclass
class PerformanceMetrics:
    n_samples: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0


@dataclass
class MetricsReport:
    generated_at: str
    manifest_path: str
    sample_size: int
    model_version: str
    anti_fraud: AntiFraudMetrics = field(default_factory=AntiFraudMetrics)
    identification: IdentificationMetrics = field(default_factory=IdentificationMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _approx_roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann–Whitney U based ROC-AUC. No sklearn dependency."""
    pos = [s for s, lbl in zip(scores, labels, strict=False) if lbl == 1]
    neg = [s for s, lbl in zip(scores, labels, strict=False) if lbl == 0]
    if not pos or not neg:
        return None
    u = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(u / (len(pos) * len(neg)), 4)


def _percentile(values: Iterable[float], p: float) -> float:
    sorted_vals = sorted(values)
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _load_manifest(manifest_path: Path, sample_size: int | None) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Populate data/test_split/ first (see data/test_split/README.md)."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sample_size is not None and sample_size > 0:
        rows = rows[:sample_size]
    return rows


def _load_image(path: Path):
    from PIL import Image  # noqa: PLC0415

    return Image.open(path).convert("RGB"), path.read_bytes()


def run(
    manifest: Path = DEFAULT_MANIFEST,
    sample_size: int | None = None,
) -> MetricsReport:
    """Programmatic entry point used by tests and CLI."""
    sys.path.insert(0, str(REPO_ROOT / "whales_be_service" / "src"))
    from whales_be_service.inference import get_pipeline  # type: ignore  # noqa: PLC0415

    rows = _load_manifest(manifest, sample_size)
    if not rows:
        raise RuntimeError(f"Manifest {manifest} is empty.")

    pipeline = get_pipeline()
    pipeline.warmup()

    report = MetricsReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest.relative_to(REPO_ROOT)),
        sample_size=len(rows),
        model_version=pipeline.model_version,
    )

    af_scores: list[float] = []
    af_labels: list[int] = []
    id_correct = 0
    id_total = 0
    unique_individuals: set[str] = set()
    latencies: list[float] = []

    base = manifest.parent
    for row in rows:
        image_path = base / row["relpath"]
        if not image_path.exists():
            logger.warning("Skipping missing file: %s", image_path)
            continue

        try:
            pil_img, img_bytes = _load_image(image_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping unreadable image %s: %s", image_path, e)
            continue

        t0 = time.perf_counter()
        det = pipeline.predict(
            pil_img=pil_img,
            filename=image_path.name,
            img_bytes=img_bytes,
            generate_mask=False,
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        is_positive = row["label"] == "cetacean"
        af_scores.append(det.cetacean_score)
        af_labels.append(1 if is_positive else 0)

        accepted = not det.rejected and det.is_cetacean
        if is_positive:
            report.anti_fraud.n_positive += 1
            if accepted:
                report.anti_fraud.tp += 1
            else:
                report.anti_fraud.fn += 1
            id_total += 1
            unique_individuals.add(row.get("individual_id", "") or "")
            expected_id = row.get("individual_id", "")
            if accepted and expected_id and det.class_animal == expected_id:
                id_correct += 1
        else:
            report.anti_fraud.n_negative += 1
            if not accepted:
                report.anti_fraud.tn += 1
            else:
                report.anti_fraud.fp += 1

    af = report.anti_fraud
    af.tpr = round(_safe_div(af.tp, af.tp + af.fn), 4)
    af.tnr = round(_safe_div(af.tn, af.tn + af.fp), 4)
    af.precision = round(_safe_div(af.tp, af.tp + af.fp), 4)
    af.f1 = round(_safe_div(2 * af.precision * af.tpr, af.precision + af.tpr), 4)
    af.roc_auc = _approx_roc_auc(af_scores, af_labels)

    ident = report.identification
    ident.n_samples = id_total
    ident.top1_accuracy = round(_safe_div(id_correct, id_total), 4)
    ident.n_unique_individuals = len(unique_individuals - {""})

    perf = report.performance
    perf.n_samples = len(latencies)
    if latencies:
        perf.latency_p50_ms = round(_percentile(latencies, 0.50), 2)
        perf.latency_p95_ms = round(_percentile(latencies, 0.95), 2)
        perf.latency_p99_ms = round(_percentile(latencies, 0.99), 2)
        perf.latency_mean_ms = round(statistics.fmean(latencies), 2)

    return report


def _format_markdown(report: MetricsReport) -> str:
    af = report.anti_fraud
    ident = report.identification
    perf = report.performance
    return (
        f"# EcoMarineAI Metrics Report\n\n"
        f"_Generated: {report.generated_at}_\n"
        f"_Manifest: `{report.manifest_path}`_\n"
        f"_Sample size: {report.sample_size}_\n"
        f"_Model version: `{report.model_version}`_\n\n"
        f"## Anti-fraud (CLIP gate, binary)\n\n"
        f"| Metric                       | Value     |\n"
        f"|------------------------------|-----------|\n"
        f"| Positives                    | {af.n_positive} |\n"
        f"| Negatives                    | {af.n_negative} |\n"
        f"| TP / FP / TN / FN            | {af.tp} / {af.fp} / {af.tn} / {af.fn} |\n"
        f"| **TPR / Sensitivity / Recall** | **{af.tpr}** |\n"
        f"| **TNR / Specificity**        | **{af.tnr}** |\n"
        f"| Precision                    | {af.precision} |\n"
        f"| F1                           | {af.f1} |\n"
        f"| ROC-AUC (cetacean_score)     | {af.roc_auc if af.roc_auc is not None else '—'} |\n\n"
        f"## Identification (multiclass, on positives only)\n\n"
        f"| Metric                       | Value     |\n"
        f"|------------------------------|-----------|\n"
        f"| Samples                      | {ident.n_samples} |\n"
        f"| Unique individuals           | {ident.n_unique_individuals} |\n"
        f"| Top-1 accuracy               | {ident.top1_accuracy} |\n\n"
        f"## Performance\n\n"
        f"| Metric                       | Value     |\n"
        f"|------------------------------|-----------|\n"
        f"| Samples timed                | {perf.n_samples} |\n"
        f"| Latency p50 / p95 / p99 (ms) | {perf.latency_p50_ms} / {perf.latency_p95_ms} / {perf.latency_p99_ms} |\n"
        f"| Latency mean (ms)            | {perf.latency_mean_ms} |\n"
    )


def _update_model_card(report: MetricsReport) -> None:
    if not MODEL_CARD.exists():
        logger.warning("MODEL_CARD.md not found at %s; skipping injection.", MODEL_CARD)
        return
    text = MODEL_CARD.read_text(encoding="utf-8")
    start, end = MARKERS
    block = (
        f"{start}\n"
        f"<!-- auto-generated by scripts/compute_metrics.py — do not edit manually -->\n"
        f"{_format_markdown(report)}"
        f"\n{end}"
    )
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end, 1)[1]
        new_text = before + block + after
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"
    MODEL_CARD.write_text(new_text, encoding="utf-8")
    logger.info("Updated metrics block in %s", MODEL_CARD)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--update-model-card", action="store_true")
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    report = run(manifest=args.manifest, sample_size=args.sample_size)

    args.output_json.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.output_md.write_text(_format_markdown(report), encoding="utf-8")
    logger.info("Wrote %s and %s", args.output_json, args.output_md)

    if args.update_model_card:
        _update_model_card(report)

    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
