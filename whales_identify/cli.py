"""Command-line interface for biologists.

Three subcommands::

    whales-cli predict <image>           # one image → human-readable report
    whales-cli batch <dir> [--csv out]   # directory → CSV summary
    whales-cli verify <image>            # anti-fraud only (yes/no)

Designed for non-developers: no Python, just paths. The CLI reuses the
exact same ``InferencePipeline`` that the FastAPI service uses, so results
match what the web UI shows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_BE_SRC = REPO_ROOT / "whales_be_service" / "src"
if _BE_SRC.exists() and str(_BE_SRC) not in sys.path:
    sys.path.insert(0, str(_BE_SRC))


def _load_image(path: Path):
    from PIL import Image  # noqa: PLC0415

    return Image.open(path).convert("RGB"), path.read_bytes()


def _get_pipeline():
    from whales_be_service.inference import get_pipeline  # type: ignore

    return get_pipeline()


def _format_human(det) -> str:
    if det.rejected and det.rejection_reason == "not_a_marine_mammal":
        return (
            f"❌ {det.image_ind}: НЕ морское млекопитающее "
            f"(cetacean_score={det.cetacean_score:.2f})"
        )
    if det.rejected and det.rejection_reason == "low_confidence":
        return (
            f"⚠️  {det.image_ind}: возможно {det.id_animal}, "
            f"но уверенность низкая ({det.probability:.2f})"
        )
    return (
        f"✅ {det.image_ind}: {det.id_animal} "
        f"(ID {det.class_animal or '—'}, confidence {det.probability:.2f}, "
        f"cetacean_score {det.cetacean_score:.2f})"
    )


def cmd_predict(args: argparse.Namespace) -> int:
    pipeline = _get_pipeline()
    pipeline.warmup()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: file not found: {image_path}", file=sys.stderr)
        return 2
    pil, raw = _load_image(image_path)
    det = pipeline.predict(pil_img=pil, filename=image_path.name, img_bytes=raw, generate_mask=False)
    if args.json:
        print(json.dumps(det.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(_format_human(det))
    return 0 if not det.rejected else 1


def cmd_batch(args: argparse.Namespace) -> int:
    pipeline = _get_pipeline()
    pipeline.warmup()
    src = Path(args.directory)
    if not src.is_dir():
        print(f"ERROR: not a directory: {src}", file=sys.stderr)
        return 2

    images = sorted(
        p
        for p in src.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not images:
        print(f"WARN: no images found under {src}")
        return 0

    rows: list[dict] = []
    for img_path in images:
        try:
            pil, raw = _load_image(img_path)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {img_path}: {e}", file=sys.stderr)
            continue
        det = pipeline.predict(
            pil_img=pil, filename=img_path.name, img_bytes=raw, generate_mask=False
        )
        rows.append(
            {
                "filename": str(img_path.relative_to(src)),
                "is_cetacean": det.is_cetacean,
                "rejected": det.rejected,
                "rejection_reason": det.rejection_reason or "",
                "species": det.id_animal,
                "individual_id": det.class_animal,
                "confidence": det.probability,
                "cetacean_score": det.cetacean_score,
            }
        )
        print(_format_human(det))

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    pipeline = _get_pipeline()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: file not found: {image_path}", file=sys.stderr)
        return 2
    pil, _ = _load_image(image_path)
    gate = pipeline.anti_fraud.score(pil)
    if gate.is_cetacean:
        print(f"ACCEPTED: cetacean_score={gate.positive_score:.4f}")
        return 0
    print(f"REJECTED: not_a_marine_mammal (cetacean_score={gate.positive_score:.4f})")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whales-cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Identify one image")
    p_predict.add_argument("image")
    p_predict.add_argument("--json", action="store_true", help="output raw JSON")
    p_predict.set_defaults(func=cmd_predict)

    p_batch = sub.add_parser("batch", help="Process a directory of images")
    p_batch.add_argument("directory")
    p_batch.add_argument("--csv", help="optional CSV output path")
    p_batch.set_defaults(func=cmd_batch)

    p_verify = sub.add_parser("verify", help="Check whether an image is a cetacean")
    p_verify.add_argument("image")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
