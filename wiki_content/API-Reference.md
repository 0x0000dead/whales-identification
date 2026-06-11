# API Reference

Complete REST API documentation for the Whales Identification (EcoMarineAI) backend.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [POST /v1/predict-single](#post-v1predict-single)
  - [POST /v1/predict-batch](#post-v1predict-batch)
  - [GET /health](#get-health)
  - [GET /metrics](#get-metrics)
  - [GET /v1/drift-stats](#get-v1drift-stats)
- [Detection schema](#detection-schema)
- [Rate limiting](#rate-limiting)
- [Client examples](#client-examples)

---

## Base URL

```
http://localhost:8000
```

When the stack is started via `docker compose up --build`, the backend is
published on port `8000` of the host. OpenAPI schema is auto-served at
`GET /docs` (Swagger UI) and `GET /redoc` (ReDoc).

**Versioning:** primary endpoints live under the `/v1/` prefix.
`POST /predict-single` and `POST /predict-batch` (without the prefix) are
backwards-compatible aliases that delegate to the v1 versions.

---

## Authentication

No authentication is required. The API applies per-IP rate limiting
(see [Rate limiting](#rate-limiting)).

---

## Endpoints

### POST /v1/predict-single

Identify a single image. Returns a `Detection` object regardless of
accept/reject — `rejected: true` is still `HTTP 200`, because a rejection is a
successful classification ("this is not a marine mammal").

#### Request

- `Content-Type: multipart/form-data`
- Field `file`: one image (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`)

#### cURL example

```bash
curl -X POST \
  -F 'file=@whale.jpg;type=image/jpeg' \
  http://localhost:8000/v1/predict-single
```

#### Response (200 — cetacean accepted)

```json
{
  "image_ind": "whale.jpg",
  "bbox": [0, 0, 512, 341],
  "class_animal": "1a71fbb72250",
  "id_animal": "humpback_whale",
  "probability": 0.847,
  "mask": "iVBORw0KGgoAAAANS...",
  "is_cetacean": true,
  "cetacean_score": 0.993,
  "rejected": false,
  "rejection_reason": null,
  "model_version": "effb4-arcface-v1",
  "candidates": [
    {"class_animal": "abc456def789", "id_animal": "humpback_whale", "probability": 0.543},
    {"class_animal": "cafe0987ba54", "id_animal": "fin_whale", "probability": 0.271}
  ]
}
```

`mask` is a base64-encoded PNG with the background removed (value shortened in
the example above); it can be `null` when mask generation is skipped.

#### Response (200 — anti-fraud rejection)

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
  "model_version": "effb4-arcface-v1",
  "candidates": []
}
```

#### Error responses

| Code | Condition                                | Example body                                        |
|------|------------------------------------------|------------------------------------------------------|
| 415  | Missing / non-image content type         | `{"detail": "Только изображения."}`                 |
| 400  | Empty upload                             | `{"detail": "Пустой файл."}`                        |
| 415  | Payload cannot be decoded as an image    | `{"detail": "Не удалось распознать изображение."}`  |
| 429  | Rate limit exceeded (60 req / 60 s / IP) | `{"detail": "Превышен лимит запросов. Повторите позже."}` |

---

### POST /v1/predict-batch

Identify every image inside a ZIP archive. Returns a list of `Detection`
objects (one per readable image).

#### Request

- `Content-Type: multipart/form-data`
- Field `archive`: one ZIP file (`application/zip` or `application/x-zip-compressed`)

#### cURL example

```bash
zip batch.zip whale1.jpg whale2.jpg cat.jpg
curl -X POST \
  -F 'archive=@batch.zip;type=application/zip' \
  http://localhost:8000/v1/predict-batch
```

#### Response (200)

```json
[
  { "image_ind": "whale1.jpg", "is_cetacean": true,  "rejected": false, "...": "..." },
  { "image_ind": "whale2.jpg", "is_cetacean": true,  "rejected": false, "...": "..." },
  { "image_ind": "cat.jpg",    "is_cetacean": false, "rejected": true,
    "rejection_reason": "not_a_marine_mammal", "...": "..." }
]
```

#### Notes

- Batch mode skips the `mask` field by default (background removal is slow).
- Corrupted / unreadable archive entries are skipped, not reported as errors.
- The whole batch counts as a single request for rate limiting.

#### Error responses

| Code | Condition            | Example body                                   |
|------|----------------------|-------------------------------------------------|
| 415  | Non-ZIP content type | `{"detail": "Ожидается ZIP-архив."}`           |
| 400  | Malformed ZIP        | `{"detail": "Не удаётся распаковать архив."}` |
| 429  | Rate limit           | `{"detail": "Превышен лимит запросов..."}`    |

---

### GET /health

Liveness probe. `device` reports the inference device — `cuda:0` when a GPU
is passed through (e.g. via `docker-compose.gpu.yml`), otherwise `cpu`.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "device": "cpu"}
```

---

### GET /metrics

Prometheus-compatible plain-text metrics: `uptime_seconds`,
`availability_percent`, `requests_total`, `errors_total`, `predictions_total`,
`rejections_total`, `rejections_by_reason{...}`, `latency_avg_ms`,
`cetacean_score_avg`.

---

### GET /v1/drift-stats

Rolling-window summary of the CLIP `cetacean_score` values seen by the
service — a lightweight data-drift signal.

```json
{
  "n": 5,
  "alarms_total": 0,
  "score_mean": 0.2111,
  "score_std": 0.3947,
  "probability_mean": 0.0151
}
```

---

## Detection schema

```python
class Detection(BaseModel):
    image_ind: str                                    # filename or ZIP entry
    bbox: list[int]                                   # [x1, y1, x2, y2]
    class_animal: str                                 # 12-hex individual_id, "" on reject
    id_animal: str                                    # species name (snake_case) or "unknown"
    probability: float                                # 0.0–1.0 identification confidence
    mask: str | None = None                           # base64 PNG, background removed
    is_cetacean: bool = True                          # CLIP anti-fraud gate decision
    cetacean_score: float                             # gate positive score, 0.0–1.0
    rejected: bool = False                            # gate or low-confidence fired
    rejection_reason: Literal[
        "not_a_marine_mammal", "low_confidence", "corrupted_image"
    ] | None = None
    model_version: str = "effb4-arcface-v1"
    candidates: list[Candidate] = []                  # top-k alternative identities


class Candidate(BaseModel):
    class_animal: str                                 # 12-hex individual_id
    id_animal: str                                    # species name
    probability: float                                # 0.0–1.0
```

Species names are returned in `snake_case` (e.g. `bottlenose_dolphin`,
`humpback_whale`).

---

## Rate limiting

60 requests per 60 seconds per client IP. Exceeding the limit returns
`HTTP 429` with a Russian-language `detail` message. A batch upload counts as
one request.

---

## Client examples

### Python

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

### JavaScript

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);

const res = await fetch("http://localhost:8000/v1/predict-single", {
  method: "POST",
  body: form,
});
const det = await res.json();
if (det.rejected) {
  console.log(`Rejected: ${det.rejection_reason}`);
} else {
  console.log(`${det.id_animal} (${(det.probability * 100).toFixed(1)}%)`);
}
```

### CLI (no code)

```bash
cd whales_be_service && poetry install
poetry run python -m whales_identify predict /path/to/whale.jpg
poetry run python -m whales_identify batch /path/to/dir/ --csv report.csv
poetry run python -m whales_identify verify /path/to/image.png
```
