#!/usr/bin/env python3
"""Populate ``data/test_split/`` from Kaggle source datasets.

Workflow (requires ``kaggle`` CLI configured with ``~/.kaggle/kaggle.json``):

1. Download 30 positive whale/dolphin images from the Happy Whale competition
   (``kaggle competitions download happy-whale-and-dolphin -f train_images/<id>.jpg``).
2. Download 30 negative images from ``rahmasleam/intel-image-dataset`` — a
   generic scene classification set (buildings / forest / glacier / mountain /
   sea / street). Each image is non-cetacean by construction.
3. Resize each image to longest side 512 px, JPEG quality 85 — keeps the repo
   under ~10 MB total while preserving enough detail for CLIP.
4. Regenerate ``data/test_split/manifest.csv``.

Re-running is idempotent: existing files are overwritten.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess  # nosec B404
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SPLIT = REPO_ROOT / "data" / "test_split"
POSITIVES = TEST_SPLIT / "positives"
NEGATIVES = TEST_SPLIT / "negatives"
MANIFEST = TEST_SPLIT / "manifest.csv"

HAPPY_WHALE_COMP = "happy-whale-and-dolphin"
INTEL_DATASET = "rahmasleam/intel-image-dataset"
MAX_SIDE = 1024  # keeps the original-ish aspect & enough detail for Laplacian
JPEG_QUALITY = 85
POSITIVES_PER_SPECIES = 10   # 10 species × 10 = 100 positives
NEGATIVES_PER_CLASS = 17     # 6 classes × 17 ≈ 102 negatives


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)  # nosec B603 - we control the args


def _download_train_csv(workdir: Path) -> Path:
    zip_path = workdir / "train.csv.zip"
    if not zip_path.exists():
        _run(
            ["kaggle", "competitions", "download", HAPPY_WHALE_COMP, "-f", "train.csv", "-p", str(workdir)],
        )
    csv_path = workdir / "train.csv"
    if not csv_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(workdir)
    return csv_path


def _pick_positives(
    train_csv: Path,
    n_per_species: int = POSITIVES_PER_SPECIES,
    n_species: int = 10,
) -> list[tuple[str, str, str]]:
    random.seed(42)
    by_species: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with train_csv.open() as f:
        for row in csv.DictReader(f):
            by_species[row["species"]].append((row["image"], row["individual_id"]))

    sample: list[tuple[str, str, str]] = []
    for species, rows in sorted(by_species.items())[:n_species]:
        picked = random.sample(rows, min(n_per_species, len(rows)))  # nosec B311 - deterministic test split, not crypto
        sample.extend((img, species, iid) for img, iid in picked)
    return sample


def _download_positive_images(sample: list[tuple[str, str, str]], workdir: Path) -> None:
    dest = workdir / "positives"
    dest.mkdir(parents=True, exist_ok=True)
    for img, _, _ in sample:
        if (dest / img).exists():
            continue
        _run(
            [
                "kaggle", "competitions", "download", HAPPY_WHALE_COMP,
                "-f", f"train_images/{img}", "-p", str(dest), "--quiet",
            ]
        )
        # Kaggle wraps single-file downloads in a zip named after the image
        zip_path = dest / f"{img}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
            zip_path.unlink()


def _download_negatives(workdir: Path) -> list[tuple[str, str]]:
    dest = workdir / "intel"
    if not dest.exists():
        dest.mkdir(parents=True)
        _run(
            ["kaggle", "datasets", "download", INTEL_DATASET, "--unzip", "-p", str(dest)],
        )
    root = next((p for p in dest.rglob("Intel Image Dataset") if p.is_dir()), None)
    if root is None:
        raise RuntimeError("Intel Image Dataset directory not found after download")

    random.seed(42)
    picked: list[tuple[str, str]] = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        jpgs = sorted(cls_dir.glob("*.jpg"))
        for img in random.sample(jpgs, min(NEGATIVES_PER_CLASS, len(jpgs))):  # nosec B311 - deterministic test split, not crypto
            new_name = f"{cls_dir.name}_{img.name}"
            picked.append((str(img), new_name))
    return picked


def _resize_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        img.save(dst, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def _write_manifest(positives: list[tuple[str, str, str]], negatives: list[tuple[str, str]]) -> None:
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relpath", "label", "individual_id", "species", "source", "license", "split"])
        for img, species, iid in positives:
            writer.writerow([f"positives/{img}", "cetacean", iid, species, "happywhale", "CC-BY-NC-4.0", "test"])
        for _, new_name in negatives:
            cls = new_name.split("_")[0]
            writer.writerow([f"negatives/{new_name}", "non_cetacean", "", cls, "intel_image_dataset", "MIT", "test"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/kaggle_hw"))  # nosec B108 - dev-only default; CI uses --workdir explicitly
    args = parser.parse_args()

    if not shutil.which("kaggle"):
        print("ERROR: kaggle CLI not found. Install with `pip install kaggle`.", file=sys.stderr)
        return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    POSITIVES.mkdir(parents=True, exist_ok=True)
    NEGATIVES.mkdir(parents=True, exist_ok=True)

    print("→ Downloading Happy Whale train.csv...")
    train_csv = _download_train_csv(args.workdir)
    sample = _pick_positives(train_csv)
    print(f"  picked {len(sample)} positive rows across {len({s for _, s, _ in sample})} species")

    print("→ Downloading positive images from Happy Whale...")
    _download_positive_images(sample, args.workdir)

    print("→ Downloading negative images from Intel Image Dataset...")
    negatives = _download_negatives(args.workdir)
    print(f"  picked {len(negatives)} negatives across {len({n.split('_')[0] for _, n in negatives})} classes")

    print(f"→ Resizing + copying into {TEST_SPLIT}...")
    for img, _, _ in sample:
        src = args.workdir / "positives" / img
        if not src.exists():
            print(f"  WARN: missing {src}")
            continue
        _resize_copy(src, POSITIVES / img)

    for src_str, new_name in negatives:
        _resize_copy(Path(src_str), NEGATIVES / new_name)

    _write_manifest(sample, negatives)
    print(f"→ Manifest written: {MANIFEST}")
    print(f"→ Total: {len(list(POSITIVES.glob('*.jpg')))} positives, {len(list(NEGATIVES.glob('*.jpg')))} negatives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
