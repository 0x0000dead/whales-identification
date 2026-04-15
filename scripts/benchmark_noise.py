#!/usr/bin/env python3
"""Measure how the anti-fraud gate holds up under image noise.

For every positive in ``data/test_split/positives/`` we generate three noisy
variants and push each through the pipeline:

1. **Gaussian noise** (σ=25) — simulates sensor noise / low-light.
2. **Low JPEG quality** (q=20) — simulates aggressive transcoding.
3. **Motion blur** (radius=4) — simulates handheld shake / fast movement.

A ТЗ-compliant system should not lose more than 20% of its baseline accept
rate on these perturbations.

Outputs ``reports/NOISE_ROBUSTNESS.md`` + ``reports/noise_robustness.json``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SPLIT = REPO_ROOT / "data" / "test_split"
OUT_JSON = REPO_ROOT / "reports" / "noise_robustness.json"
OUT_MD = REPO_ROOT / "reports" / "NOISE_ROBUSTNESS.md"


def _gaussian_noise(img, sigma: float = 25.0):
    import numpy as np
    from PIL import Image

    arr = np.array(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def _low_jpeg(img, quality: int = 20):
    from PIL import Image

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _motion_blur(img, radius: int = 4):
    from PIL import ImageFilter

    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius))


VARIANTS: dict[str, Callable] = {
    "clean": lambda x: x.convert("RGB"),
    "gaussian_sigma25": _gaussian_noise,
    "jpeg_q20": _low_jpeg,
    "blur_r4": _motion_blur,
}


def _load_positives(manifest: Path) -> list[Path]:
    with manifest.open() as f:
        return [
            TEST_SPLIT / row["relpath"]
            for row in csv.DictReader(f)
            if row["label"] == "cetacean"
        ]


def run() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "whales_be_service" / "src"))
    from PIL import Image

    from whales_be_service.inference import get_pipeline

    pipeline = get_pipeline()
    pipeline.warmup()

    positives = _load_positives(TEST_SPLIT / "manifest.csv")
    if not positives:
        raise RuntimeError("No positive images found; populate data/test_split/positives/ first")
    positives = [p for p in positives if p.exists()]

    results: dict[str, dict] = {}
    for variant_name, transform in VARIANTS.items():
        accepted = 0
        total = 0
        scores: list[float] = []
        for path in positives:
            try:
                img = Image.open(path).convert("RGB")
                transformed = transform(img)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {path.name} ({variant_name}): {e}")
                continue
            det = pipeline.predict(
                pil_img=transformed,
                filename=f"{variant_name}_{path.name}",
                img_bytes=None,
                generate_mask=False,
            )
            total += 1
            scores.append(det.cetacean_score)
            if det.is_cetacean:
                accepted += 1
        accept_rate = accepted / max(total, 1)
        results[variant_name] = {
            "accepted": accepted,
            "total": total,
            "accept_rate": round(accept_rate, 4),
            "mean_cetacean_score": round(sum(scores) / max(len(scores), 1), 4),
        }

    baseline = results.get("clean", {}).get("accept_rate", 0.0) or 1e-9
    for _variant, data in results.items():
        drop = (baseline - data["accept_rate"]) / baseline
        data["drop_vs_clean"] = round(drop, 4)
        data["passes_20pct_target"] = drop <= 0.20
    return results


def _write_markdown(report: dict) -> str:
    lines = [
        "# Noise robustness benchmark",
        "",
        "Measures accept rate of the CLIP anti-fraud gate when positives are",
        "corrupted by three realistic noise sources. The ТЗ (Параметр 4) demands",
        "that accuracy drop by ≤ 20% under such conditions.",
        "",
        "| Variant | Accepted / Total | Accept rate | Mean score | Drop vs clean | ≤ 20% target |",
        "|---|---|---|---|---|:---:|",
    ]
    for name, data in report.items():
        accept_rate = f"{data['accept_rate']:.4f}"
        drop_pct = f"{data['drop_vs_clean']*100:+.1f} %"
        status = "✓" if data["passes_20pct_target"] else "✗"
        lines.append(
            f"| `{name}` | {data['accepted']}/{data['total']} | {accept_rate} | "
            f"{data['mean_cetacean_score']:.4f} | {drop_pct} | {status} |"
        )
    lines += [
        "",
        "## Variant recipes",
        "- `clean`: untouched RGB image",
        "- `gaussian_sigma25`: per-pixel N(0, 25²) noise",
        "- `jpeg_q20`: re-encoded as JPEG quality 20",
        "- `blur_r4`: PIL Gaussian blur radius 4",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np

    np.random.seed(args.seed)

    report = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    OUT_MD.write_text(_write_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
