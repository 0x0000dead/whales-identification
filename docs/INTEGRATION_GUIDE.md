# Integration guide

The service exposes three integration surfaces:

1. **REST API** — language-agnostic, any HTTP client works.
2. **Python library** (`whales_identify/cli.py`) — for custom scripts.
3. **Integration sinks** (`integrations/`) — wire predictions into external data stores.

This guide walks through concrete examples for each.

---

## 1. REST API — curl, Python, JavaScript, R

### curl

```bash
curl -X POST \
    -F 'file=@whale.jpg;type=image/jpeg' \
    http://localhost:8000/v1/predict-single
```

### Python (`requests`)

```python
import requests

with open("whale.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/predict-single",
        files={"file": ("whale.jpg", f, "image/jpeg")},
        timeout=30,
    )
resp.raise_for_status()
det = resp.json()
print(f"{det['id_animal']} — {det['probability']:.2%}")
```

### JavaScript (`fetch`)

```js
const fd = new FormData();
fd.append("file", fileInput.files[0]);
const r = await fetch("http://localhost:8000/v1/predict-single", {
  method: "POST",
  body: fd,
});
const det = await r.json();
if (det.rejected) {
  console.log(`Rejected: ${det.rejection_reason}`);
} else {
  console.log(`${det.id_animal} @ ${(det.probability * 100).toFixed(1)}%`);
}
```

### R (`httr2`)

```r
library(httr2)
req <- request("http://localhost:8000/v1/predict-single") |>
  req_body_multipart(file = curl::form_file("whale.jpg", type="image/jpeg"))
resp <- req_perform(req)
det <- resp_body_json(resp)
print(det$id_animal)
```

---

## 2. CLI for biologists (zero Python knowledge)

```bash
# One image
python -m whales_identify predict /photos/survey_01.jpg

# Batch → CSV report
python -m whales_identify batch /photos/survey/ --csv survey.csv

# Only the anti-fraud gate (fast sanity check)
python -m whales_identify verify /photos/random.jpg
```

Sample `survey.csv`:

```csv
filename,is_cetacean,rejected,rejection_reason,species,individual_id,confidence,cetacean_score
001.jpg,True,False,,humpback_whale,1a71fbb72250,0.72,0.99
002.jpg,False,True,not_a_marine_mammal,,,0.0,0.07
003.jpg,True,False,,bottlenose_dolphin,b9907151f66e,0.31,0.92
```

---

## 3. Database sinks

### 3.1 SQLite (zero-dep, local)

```bash
python3 integrations/sqlite_sink.py \
    --directory /photos/survey \
    --db observations.sqlite
```

Resulting schema:

```sql
CREATE TABLE detections (
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
    bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
    model_version TEXT NOT NULL
);
```

Query examples:

```sql
-- Species histogram
SELECT id_animal, COUNT(*) FROM detections
WHERE NOT rejected
GROUP BY id_animal ORDER BY 2 DESC;

-- Hour-by-hour activity
SELECT strftime('%Y-%m-%d %H', created_at) AS hr, COUNT(*)
FROM detections GROUP BY hr;

-- Low-confidence cases for manual review
SELECT filename, id_animal, probability
FROM detections
WHERE NOT rejected AND probability < 0.3
ORDER BY probability;
```

### 3.2 PostgreSQL (production)

```bash
pip install 'psycopg[binary]>=3.1'
python3 integrations/postgres_sink.py \
    --directory /photos/survey \
    --dsn 'postgresql://user:pass@db.example.com:5432/ecomarine'
```

Schema is identical to SQLite but uses `BIGSERIAL`, `TIMESTAMPTZ`, and a native `INTEGER[]` for `bbox`.

### 3.3 CSV (stdout)

Already covered by `whales-cli batch --csv`. Pipe it to anything that reads CSV.

### 3.4 Prometheus (metrics only, no per-image data)

```yaml
# prometheus.yml
scrape_configs:
  - job_name: ecomarine
    scrape_interval: 15s
    static_configs:
      - targets: ["ecomarine-backend:8000"]
    metrics_path: /metrics
```

---

## 4. Model hub integration (HuggingFace)

Model weights live on HuggingFace under `0x0000dead/ecomarineai-cetacean-effb4`. The Docker entrypoint downloads them automatically, but you can also use them directly from Python:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    repo_id="0x0000dead/ecomarineai-cetacean-effb4",
    filename="efficientnet_b4_512_fold0.ckpt",
)
```

See [ML_ARCHITECTURE.md](ML_ARCHITECTURE.md) §2 for a complete loader example.

---

## 5. Sync with biodiversity platforms

The sink modules can be extended to push observations to external biodiversity databases (GBIF, OBIS, iNaturalist). Each platform accepts Darwin Core-formatted records:

```python
record = {
    "eventDate": det["created_at"],
    "scientificName": det["id_animal"].replace("_", " "),
    "occurrenceID": f"ecomarineai-{det['class_animal']}-{det['image_ind']}",
    "basisOfRecord": "MachineObservation",
    "identifiedBy": "EcoMarineAI model v1.1",
    "identificationRemarks": f"probability={det['probability']:.4f}",
}
```

A `darwin_core_sink.py` is on the roadmap for Q3 2026 — see [ROADMAP.md](ROADMAP.md).

---

## 6. Rate limiting, retries, backoff

The service has a built-in 60 req / 60 s per-IP throttle. If you're batching thousands of images from a single client:

- Use the `/v1/predict-batch` ZIP endpoint (one HTTP call, many images).
- Add exponential backoff on HTTP 429 responses.
- Spread requests across IPs or authenticate with a reverse proxy that bumps the limit for known clients.

Example retry loop:

```python
import time, requests
from requests.adapters import HTTPAdapter, Retry

sess = requests.Session()
retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
sess.mount("http://", HTTPAdapter(max_retries=retries))
```
