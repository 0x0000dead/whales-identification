# API Changelog

All notable changes to the EcoMarineAI API are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
