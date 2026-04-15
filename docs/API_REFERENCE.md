# API Reference

**Base URL:** `http://localhost:8000` (or whatever you set via `ALLOWED_ORIGINS`).

OpenAPI schema is auto-served at `GET /docs` (Swagger UI) and `GET /redoc` (ReDoc).

---

## `GET /health`

Liveness probe. Returns `{"status": "ok"}` iff the service process is up.

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

---

## `GET /metrics`

Prometheus-compatible plain text metrics. Scrape interval 15 s is fine.

Exposed counters and gauges:

| Name                       | Type    | Meaning                                                   |
|----------------------------|---------|-----------------------------------------------------------|
| `uptime_seconds`           | counter | Seconds since process start                                |
| `availability_percent`     | gauge   | (requests − errors) / requests × 100                       |
| `requests_total`           | counter | All HTTP requests                                          |
| `errors_total`             | counter | Requests that returned ≥ 400                               |
| `predictions_total`        | counter | Successful (not rejected) predictions                      |
| `rejections_total`         | counter | Rejections (either anti-fraud or low-confidence)           |
| `rejections_by_reason{...}`| counter | Rejection breakdown by `not_a_marine_mammal`/`low_confidence` |
| `latency_avg_ms`           | gauge   | Mean HTTP latency                                          |
| `cetacean_score_avg`       | gauge   | Rolling mean of the CLIP positive score                    |

---

## `POST /v1/predict-single`

Identify a single image. Returns a `Detection` object regardless of accept/reject — `rejected: true` is still `HTTP 200`, because rejection is a successful classification ("this is not a whale").

### Request

- `Content-Type: multipart/form-data`
- Field `file`: one image (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`).

### Example

```bash
curl -X POST \
  -F 'file=@whale.jpg;type=image/jpeg' \
  http://localhost:8000/v1/predict-single
```

### Response (200 — cetacean)

```json
{
  "image_ind": "whale.jpg",
  "bbox": [0, 0, 512, 341],
  "class_animal": "a6e325d8e924",
  "id_animal": "bottlenose_dolphin",
  "probability": 0.0756,
  "mask": null,
  "is_cetacean": true,
  "cetacean_score": 0.9997,
  "rejected": false,
  "rejection_reason": null,
  "model_version": "effb4-arcface-v1"
}
```

### Response (200 — anti-fraud rejection)

```json
{
  "image_ind": "text_screenshot.png",
  "bbox": [0, 0, 800, 600],
  "class_animal": "",
  "id_animal": "unknown",
  "probability": 0.0,
  "mask": null,
  "is_cetacean": false,
  "cetacean_score": 0.08,
  "rejected": true,
  "rejection_reason": "not_a_marine_mammal",
  "model_version": "effb4-arcface-v1"
}
```

### Error responses

| Code | Condition                                 | Example body                                    |
|------|-------------------------------------------|--------------------------------------------------|
| 415  | Missing / non-image content type          | `{"detail": "Только изображения."}`             |
| 400  | Empty upload                              | `{"detail": "Пустой файл."}`                    |
| 415  | PIL can't decode the payload              | `{"detail": "Не удалось распознать изображение."}` |
| 429  | Rate-limit exceeded (60 req / 60 s / IP)  | `{"detail": "Превышен лимит запросов. Повторите позже."}` |

---

## `POST /v1/predict-batch`

Identify every image in a ZIP archive. Returns a list of `Detection` objects (one per readable image).

### Request

- `Content-Type: multipart/form-data`
- Field `archive`: one ZIP (`application/zip` or `application/x-zip-compressed`).

### Example

```bash
zip batch.zip whale1.jpg whale2.jpg cat.jpg
curl -X POST \
  -F 'archive=@batch.zip;type=application/zip' \
  http://localhost:8000/v1/predict-batch
```

### Response (200)

```json
[
  { "image_ind": "whale1.jpg", "is_cetacean": true,  "rejected": false, ...},
  { "image_ind": "whale2.jpg", "is_cetacean": true,  "rejected": false, ...},
  { "image_ind": "cat.jpg",    "is_cetacean": false, "rejected": true,
    "rejection_reason": "not_a_marine_mammal", ...}
]
```

### Notes

- Batch mode skips the `mask` field by default (rembg is slow).
- Corrupted / unreadable entries are silently dropped, not reported as errors.
- Per-image rate-limiting is **not** applied; the whole batch counts as one request.

### Error responses

| Code | Condition                       | Example body                                       |
|------|---------------------------------|-----------------------------------------------------|
| 415  | Non-ZIP content type            | `{"detail": "Ожидается ZIP-архив."}`               |
| 400  | Malformed ZIP                   | `{"detail": "Не удаётся распаковать архив."}`     |
| 429  | Rate-limit                      | `{"detail": "Превышен лимит запросов..."}`        |

---

## `GET /v1/drift-stats`

Rolling-window summary of CLIP `cetacean_score` values seen by the service. Useful as a lightweight drift signal.

```json
{
  "n": 5,
  "alarms_total": 0,
  "score_mean": 0.2111,
  "score_std": 0.3947,
  "probability_mean": 0.0151
}
```

- `n`: number of predictions in the rolling window (max 1000).
- `alarms_total`: times the window mean dropped > 10 pp below the calibrated baseline.
- Everything else is statistics over the window.

---

## Backwards-compatible aliases

`POST /predict-single` and `POST /predict-batch` (without the `/v1` prefix) delegate to the v1 versions so legacy clients don't break during upgrades.

---

## Detection schema (Pydantic)

```python
class Detection(BaseModel):
    image_ind: str                                    # filename or ZIP entry
    bbox: list[int]                                   # [x1, y1, x2, y2]
    class_animal: str                                 # 12-hex individual_id, "" on reject
    id_animal: str                                    # species name or "unknown"
    probability: float                                # 0.0–1.0 identification confidence
    mask: str | None = None                           # base64 PNG, optional
    is_cetacean: bool = True                          # CLIP gate decision
    cetacean_score: float = Field(ge=0, le=1, default=1.0)  # gate positive softmax
    rejected: bool = False                            # true if gate or low-confidence fired
    rejection_reason: Literal[
        "not_a_marine_mammal", "low_confidence", "corrupted_image"
    ] | None = None
    model_version: str = "effb4-arcface-v1"
```

All new fields (`is_cetacean` onwards) have defaults, so the response is a strict superset of the v1.0 shape; old clients continue to parse new responses without changes.

---

## Python client example

```python
import requests

with open("whale.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/v1/predict-single",
        files={"file": ("whale.jpg", f, "image/jpeg")},
        timeout=30,
    )
r.raise_for_status()
det = r.json()
if det["rejected"]:
    print(f"Rejected: {det['rejection_reason']} (score={det['cetacean_score']})")
else:
    print(f"{det['id_animal']} — {det['class_animal']} @ {det['probability']:.2%}")
```
