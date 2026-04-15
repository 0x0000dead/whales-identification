#!/usr/bin/env python3
"""Sweep the CLIP anti-fraud threshold on the in-repo test split.

Picks the smallest threshold that satisfies ``TNR >= 0.90`` (specificity
target from the TZ) while keeping ``TPR >= 0.85`` (sensitivity target).
Writes the result to::

    whales_be_service/src/whales_be_service/configs/anti_fraud_threshold.yaml

So the next service start picks up the calibrated value automatically.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "test_split" / "manifest.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "whales_be_service"
    / "src"
    / "whales_be_service"
    / "configs"
    / "anti_fraud_threshold.yaml"
)
ROC_PNG = REPO_ROOT / "DOCS" / "anti_fraud_roc.png"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("calibrate_clip")


def _load_manifest(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _score_image(gate, pil_img):
    return gate.score(pil_img).positive_score


def _sweep(scores_pos: list[float], scores_neg: list[float]) -> list[dict[str, float]]:
    results = []
    for i in range(30, 81):
        t = i / 100.0
        tp = sum(1 for s in scores_pos if s >= t)
        fn = len(scores_pos) - tp
        tn = sum(1 for s in scores_neg if s < t)
        fp = len(scores_neg) - tn
        tpr = tp / max(len(scores_pos), 1)
        tnr = tn / max(len(scores_neg), 1)
        results.append({"threshold": t, "tpr": tpr, "tnr": tnr, "tp": tp, "fn": fn, "tn": tn, "fp": fp})
    return results


def _pick_threshold(rows: list[dict[str, float]]) -> dict[str, float] | None:
    candidates = [r for r in rows if r["tnr"] >= 0.90 and r["tpr"] >= 0.85]
    if not candidates:
        return None
    candidates.sort(key=lambda r: (r["threshold"]))
    return candidates[0]


def _maybe_save_roc(rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.info("matplotlib not installed; skipping ROC plot.")
        return
    ROC_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    fpr = [1 - r["tnr"] for r in rows]
    tpr = [r["tpr"] for r in rows]
    plt.plot(fpr, tpr, marker="o", linewidth=1)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate (1 - TNR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("EcoMarineAI Anti-Fraud Gate — ROC")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=120)
    logger.info("Saved ROC plot to %s", ROC_PNG)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "whales_be_service" / "src"))
    from PIL import Image  # noqa: PLC0415

    from whales_be_service.inference.anti_fraud import AntiFraudGate  # noqa: PLC0415

    gate = AntiFraudGate()
    gate._load()  # noqa: SLF001

    rows = _load_manifest(args.manifest)
    base = args.manifest.parent
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for row in rows:
        path = base / row["relpath"]
        if not path.exists():
            logger.warning("Missing %s — skipping", path)
            continue
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            logger.warning("Cannot open %s: %s", path, e)
            continue
        score = _score_image(gate, pil)
        if row["label"] == "cetacean":
            pos_scores.append(score)
        else:
            neg_scores.append(score)

    logger.info("Scored %d positives, %d negatives", len(pos_scores), len(neg_scores))
    if not pos_scores or not neg_scores:
        logger.error(
            "Need at least one positive and one negative sample to calibrate. "
            "Populate data/test_split/ first."
        )
        return 1

    sweep = _sweep(pos_scores, neg_scores)
    chosen = _pick_threshold(sweep)
    if chosen is None:
        logger.warning(
            "No threshold satisfies TNR>=0.90 AND TPR>=0.85. "
            "Reporting best-TNR fallback."
        )
        chosen = max(sweep, key=lambda r: r["tnr"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(
            {
                "threshold": round(chosen["threshold"], 4),
                "tpr": round(chosen["tpr"], 4),
                "tnr": round(chosen["tnr"], 4),
                "n_positive": len(pos_scores),
                "n_negative": len(neg_scores),
                "calibrated_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", args.output)

    _maybe_save_roc(sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
