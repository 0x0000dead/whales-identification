# API Changelog

All notable changes to the EcoMarineAI API are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] - 2026-04-15

### Added

- **CLIP zero-shot anti-fraud gate.** Every prediction is gated by an OpenCLIP ViT-B/32 model that compares the image against 10 positive (whale/dolphin) and 14 negative (text/people/landscape/etc.) prompts. Non-cetacean images return `200 OK` with `rejected: true` and `rejection_reason: "not_a_marine_mammal"` instead of a hallucinated whale ID.
- **Real ML inference.** `detection_id()` (random mock) has been removed. The `/v1/predict-single` and `/v1/predict-batch` endpoints now route through `inference.InferencePipeline`, which loads the ViT-L/32 checkpoint lazily at FastAPI startup.
- New `Detection` fields (default-valued, additive — no breaking changes for legacy clients):
  - `is_cetacean: bool`
  - `cetacean_score: float` (CLIP positive aggregate softmax)
  - `rejected: bool`
  - `rejection_reason: "not_a_marine_mammal" | "low_confidence" | "corrupted_image" | null`
  - `model_version: string`
- New endpoint `GET /v1/drift-stats` — rolling-window mean/std/p50 of `cetacean_score`.
- `/metrics` Prometheus output now includes `rejections_total`, `rejections_by_reason`, `cetacean_score_avg`.
- CORS origins now driven by `ALLOWED_ORIGINS` env var (default: localhost dev ports).
- Lifespan startup warms up both gate and identification model.

### Changed

- `/predict-single` and `/predict-batch` now alias `/v1/*` endpoints (no behavioural change for existing callers).
- Identification CSV / weights resolved via `inference.identification.IdentificationModel` (lazy load) instead of module-level `torch.load()`.

### Removed

- `detection_id()` mock function.
- `whale_infer.py` and the unused `routers.py` empty stub.
- Hardcoded precision values from `models_config.yaml`.

## [1.0.0] - 2024-12-01

### Added

- `POST /predict-single` - Single image identification endpoint
  - Accepts multipart/form-data with image file
  - Returns Detection object with bbox, species, individual ID, probability, and mask
- `POST /predict-batch` - Batch processing via ZIP archive
  - Accepts multipart/form-data with ZIP archive
  - Returns array of Detection objects
- `GET /health` - Health check endpoint for monitoring and container orchestration
- CORS middleware allowing all origins
- Background removal (rembg) generating base64 PNG masks
- Config.yaml mapping 15,587 individual IDs to species names

### API Response Format

```json
{
  "image_ind": "filename.jpg",
  "bbox": [x, y, width, height],
  "class_animal": "individual_hex_id",
  "id_animal": "species_name",
  "probability": 0.95,
  "mask": "base64_encoded_png"
}
```

### Error Codes

- `415` - Unsupported media type (non-image for single, non-ZIP for batch)
- `400` - Bad ZIP archive (corrupted or invalid format)

## [0.1.0] - 2024-06-01

### Added

- Initial prototype API with mock predictions
- FastAPI backend with uvicorn server
- Docker containerization with docker-compose
- Frontend React application with single and batch upload
