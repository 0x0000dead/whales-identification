#!/usr/bin/env python3
"""Demonstrate linear time complexity of the EcoMarineAI pipeline.

Runs the pipeline on 10, 20, 40, and 60 images from ``data/test_split/`` and
reports total wall-clock latency. With a linear-complexity pipeline the slope
should be near-constant (≈ const × N, where const is the per-image cost).

Outputs a Markdown table to ``reports/SCALABILITY.md`` and raw JSON to
``reports/scalability_latest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SPLIT = REPO_ROOT / "data" / "test_split"
OUT_JSON = REPO_ROOT / "reports" / "scalability_latest.json"
OUT_MD = REPO_ROOT / "reports" / "SCALABILITY.md"


def _load_images(manifest: Path, limit: int):
    import csv

    from PIL import Image

    rows = []
    with manifest.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "cetacean":
                rows.append(row)
            if len(rows) >= limit:
                break
    pil_imgs = []
    for row in rows:
        path = TEST_SPLIT / row["relpath"]
        if path.exists():
            pil_imgs.append((path.name, Image.open(path).convert("RGB"), path.read_bytes()))
    return pil_imgs


def run(sizes: list[int]) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "whales_be_service" / "src"))
    from whales_be_service.inference import get_pipeline

    pipeline = get_pipeline()
    pipeline.warmup()
    manifest = TEST_SPLIT / "manifest.csv"
    biggest = max(sizes)
    pool = _load_images(manifest, biggest)
    if len(pool) < biggest:
        print(f"WARN: only {len(pool)} positives available; trimming sizes to fit")
        sizes = [s for s in sizes if s <= len(pool)]

    # Warmup pass with the first image (to hit any per-call JIT paths).
    if pool:
        pipeline.predict(pil_img=pool[0][1], filename=pool[0][0], img_bytes=pool[0][2], generate_mask=False)

    report = {"points": [], "regression": {}}
    for n in sizes:
        batch = pool[:n]
        t0 = time.perf_counter()
        for name, pil, raw in batch:
            pipeline.predict(pil_img=pil, filename=name, img_bytes=raw, generate_mask=False)
        total = time.perf_counter() - t0
        per_img = total / max(n, 1)
        report["points"].append(
            {"n": n, "total_s": round(total, 4), "per_image_s": round(per_img, 4)}
        )
        print(f"n={n:4d}  total={total:6.2f}s  per_image={per_img*1000:7.1f}ms")

    # Simple linear regression: slope ≈ per-image cost, intercept ≈ warmup.
    xs = [p["n"] for p in report["points"]]
    ys = [p["total_s"] for p in report["points"]]
    n = len(xs)
    if n >= 2:
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
        den = sum((x - mean_x) ** 2 for x in xs) or 1.0
        slope = num / den
        intercept = mean_y - slope * mean_x
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=False))
        ss_tot = sum((y - mean_y) ** 2 for y in ys) or 1.0
        r_squared = 1 - ss_res / ss_tot
        report["regression"] = {
            "slope_s_per_image": round(slope, 4),
            "intercept_s": round(intercept, 4),
            "r_squared": round(r_squared, 4),
        }
    return report


def _write_markdown(report: dict) -> str:
    lines = [
        "# Scalability benchmark",
        "",
        "Measures wall-clock latency of the EcoMarineAI pipeline on batches of",
        "N images drawn from `data/test_split/positives/`.",
        "",
        "## Results",
        "",
        "| N images | Total (s) | Per image (ms) |",
        "|---------:|----------:|---------------:|",
    ]
    for p in report["points"]:
        lines.append(f"| {p['n']} | {p['total_s']} | {p['per_image_s']*1000:.0f} |")

    reg = report.get("regression", {})
    if reg:
        lines += [
            "",
            "## Linear regression",
            "",
            f"- slope: **{reg['slope_s_per_image']} s/image** (marginal cost)",
            f"- intercept: {reg['intercept_s']} s (one-off warmup)",
            f"- R²: **{reg['r_squared']}** (1.0 = perfect linear fit)",
            "",
            "If R² ≥ 0.99 the pipeline has linear time complexity, as required",
            "by the ТЗ (Параметр 3 — Масштабируемость системы).",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30],
        help="Batch sizes to sweep (default: 5 10 20 30)",
    )
    args = parser.parse_args()

    report = run(args.sizes)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    OUT_MD.write_text(_write_markdown(report))
    print()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
