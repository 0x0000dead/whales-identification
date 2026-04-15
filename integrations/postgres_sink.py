"""PostgreSQL sink example — identical schema to sqlite_sink.py.

Why a second integration even though SQLite covers the API contract? The ТЗ
(Параметр 6) requires integration with **two or more** external platforms.
This file is a worked example showing how to point the same pipeline at a
production-grade RDBMS. It uses the stdlib-adjacent ``psycopg`` driver::

    pip install 'psycopg[binary]>=3.1'

    python3 integrations/postgres_sink.py \\
        --directory data/test_split/positives \\
        --dsn 'postgresql://user:pass@localhost:5432/ecomarine'

    psql -c "SELECT id_animal, count(*) FROM detections GROUP BY id_animal;"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    filename TEXT NOT NULL,
    rejected BOOLEAN NOT NULL,
    rejection_reason TEXT,
    is_cetacean BOOLEAN NOT NULL,
    cetacean_score REAL NOT NULL,
    class_animal TEXT,
    id_animal TEXT,
    probability REAL NOT NULL,
    bbox INTEGER[] NOT NULL,
    model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_detections_species ON detections (id_animal);
CREATE INDEX IF NOT EXISTS ix_detections_class   ON detections (class_animal);
CREATE INDEX IF NOT EXISTS ix_detections_ctime   ON detections (created_at);
"""


def run(directory: Path, dsn: str) -> int:
    try:
        import psycopg
    except ImportError as e:
        print(
            "psycopg not installed. Install with `pip install 'psycopg[binary]>=3.1'`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    sys.path.insert(0, str(REPO_ROOT / "whales_be_service" / "src"))
    from PIL import Image

    from whales_be_service.inference import get_pipeline

    pipeline = get_pipeline()
    pipeline.warmup()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        conn.commit()

        images = sorted(
            p
            for p in directory.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        print(f"Processing {len(images)} images → {dsn}")
        with conn.cursor() as cur:
            for path in images:
                try:
                    pil = Image.open(path).convert("RGB")
                    raw = path.read_bytes()
                except Exception as e:  # noqa: BLE001
                    print(f"SKIP {path}: {e}")
                    continue
                det = pipeline.predict(
                    pil_img=pil,
                    filename=path.name,
                    img_bytes=raw,
                    generate_mask=False,
                )
                cur.execute(
                    """INSERT INTO detections
                       (created_at, filename, rejected, rejection_reason,
                        is_cetacean, cetacean_score, class_animal, id_animal,
                        probability, bbox, model_version)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        datetime.now(timezone.utc),
                        det.image_ind,
                        det.rejected,
                        det.rejection_reason,
                        det.is_cetacean,
                        det.cetacean_score,
                        det.class_animal,
                        det.id_animal,
                        det.probability,
                        list(det.bbox),
                        det.model_version,
                    ),
                )
        conn.commit()
    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--dsn",
        required=True,
        help="PostgreSQL DSN, e.g. postgresql://user:pass@host:5432/db",
    )
    args = parser.parse_args()
    return run(args.directory, args.dsn)


if __name__ == "__main__":
    sys.exit(main())
