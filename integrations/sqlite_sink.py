"""SQLite sink for EcoMarineAI predictions — example integration.

Writes every ``Detection`` returned by the pipeline into a local SQLite DB.
The same schema transfers 1:1 to PostgreSQL / MySQL — only the driver import
changes. Biologists can point the CLI at a dataset and get an analysable
database without touching Python::

    python3 integrations/sqlite_sink.py \\
        --directory data/test_split/positives \\
        --db observations.sqlite

    sqlite3 observations.sqlite 'SELECT species, COUNT(*) FROM detections GROUP BY species;'
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    rejected INTEGER NOT NULL,
    rejection_reason TEXT,
    is_cetacean INTEGER NOT NULL,
    cetacean_score REAL NOT NULL,
    class_animal TEXT,
    id_animal TEXT,
    probability REAL NOT NULL,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    model_version TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_detections_species ON detections(id_animal);
CREATE INDEX IF NOT EXISTS ix_detections_class  ON detections(class_animal);
CREATE INDEX IF NOT EXISTS ix_detections_ctime  ON detections(created_at);
"""


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _insert(conn: sqlite3.Connection, det) -> None:
    conn.execute(
        """INSERT INTO detections
           (created_at, filename, rejected, rejection_reason, is_cetacean,
            cetacean_score, class_animal, id_animal, probability,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, model_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            det.image_ind,
            int(det.rejected),
            det.rejection_reason,
            int(det.is_cetacean),
            det.cetacean_score,
            det.class_animal,
            det.id_animal,
            det.probability,
            det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
            det.model_version,
        ),
    )


def run(directory: Path, db_path: Path) -> int:
    sys.path.insert(0, str(REPO_ROOT / "whales_be_service" / "src"))
    from PIL import Image

    from whales_be_service.inference import get_pipeline

    pipeline = get_pipeline()
    pipeline.warmup()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _init_db(db_path)

    images = sorted(
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    print(f"Processing {len(images)} images from {directory} → {db_path}")
    n = 0
    for path in images:
        try:
            pil = Image.open(path).convert("RGB")
            raw = path.read_bytes()
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {path}: {e}")
            continue
        det = pipeline.predict(
            pil_img=pil, filename=path.name, img_bytes=raw, generate_mask=False
        )
        _insert(conn, det)
        n += 1
    conn.commit()
    conn.close()
    print(f"Inserted {n} rows into {db_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("observations.sqlite"))
    args = parser.parse_args()
    return run(args.directory, args.db)


if __name__ == "__main__":
    sys.exit(main())
