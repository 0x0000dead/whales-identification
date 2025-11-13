# API Reference

Complete documentation for the Whales Identification REST API.

---

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [POST /predict-single](#post-predict-single)
  - [POST /predict-batch](#post-predict-batch)
- [Response Models](#response-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Code Examples](#code-examples)

---

## Base URL

```
http://localhost:8000
```

**Production (if deployed):**
```
https://your-domain.com
```

---

## Authentication

Currently, the API does **not require authentication**.

**Future versions** may implement:
- API Keys
- JWT tokens
- OAuth 2.0

---

## Endpoints

### POST /predict-single

Predict species and individual ID for a single whale image.

#### Request

**Endpoint:**
```
POST /predict-single
```

**Headers:**
```
Content-Type: multipart/form-data
```

**Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | ✅ Yes | Image file (JPG, PNG, JPEG) |

**Supported formats:**
- `.jpg`, `.jpeg`, `.png`
- Max file size: 10 MB (default)
- Recommended resolution: 1920×1080 or higher

#### Response

**Success (200 OK):**

```json
{
  "image_ind": "whale_001.jpg",
  "bbox": [120, 180, 400, 300],
  "class_animal": "a1b2c3d4e5f6",
  "id_animal": "Humpback Whale",
  "probability": 0.953,
  "mask": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `image_ind` | string | Original filename |
| `bbox` | array[int] | Bounding box [x, y, width, height] |
| `class_animal` | string | Individual whale ID (hex-like) |
| `id_animal` | string | Species name (e.g., "Humpback Whale") |
| `probability` | float | Confidence score (0.0-1.0) |
| `mask` | string | Base64-encoded PNG with background removed |

#### Error Responses

**400 Bad Request:**
```json
{
  "detail": "Unsupported media type. Allowed: image/jpeg, image/png"
}
```

**422 Unprocessable Entity:**
```json
{
  "detail": "No file provided"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Model inference failed"
}
```

#### cURL Example

```bash
curl -X POST "http://localhost:8000/predict-single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/whale_image.jpg"
```

#### Python Example

```python
import requests

# Single image prediction
url = "http://localhost:8000/predict-single"

with open("whale_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Species: {result['id_animal']}")
    print(f"Individual ID: {result['class_animal']}")
    print(f"Confidence: {result['probability']:.2%}")

    # Save mask image
    import base64
    mask_data = base64.b64decode(result['mask'])
    with open("whale_mask.png", "wb") as mask_file:
        mask_file.write(mask_data)
else:
    print(f"Error {response.status_code}: {response.text}")
```

#### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict-single', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log('Species:', data.id_animal);
    console.log('Confidence:', data.probability);

    // Display mask
    const maskImg = document.createElement('img');
    maskImg.src = `data:image/png;base64,${data.mask}`;
    document.body.appendChild(maskImg);
  })
  .catch(error => console.error('Error:', error));
```

---

### POST /predict-batch

Process multiple images in a ZIP archive.

#### Request

**Endpoint:**
```
POST /predict-batch
```

**Headers:**
```
Content-Type: application/zip
```

**Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | ✅ Yes | ZIP archive containing images |

**ZIP Structure:**
```
batch_images.zip
├── whale_001.jpg
├── whale_002.png
└── whale_003.jpeg
```

**Requirements:**
- ZIP file size: Max 50 MB
- Individual images: JPG, PNG, JPEG
- Max images per batch: 100 (default)

#### Response

**Success (200 OK):**

```json
{
  "results": [
    {
      "image_ind": "whale_001.jpg",
      "bbox": [120, 180, 400, 300],
      "class_animal": "a1b2c3d4e5f6",
      "id_animal": "Humpback Whale",
      "probability": 0.953,
      "mask": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "image_ind": "whale_002.png",
      "bbox": [80, 120, 350, 280],
      "class_animal": "b2c3d4e5f6a7",
      "id_animal": "Blue Whale",
      "probability": 0.887,
      "mask": "iVBORw0KGgoAAAANSUhEUgBB..."
    }
  ],
  "total_processed": 2,
  "processing_time_seconds": 5.34
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Array of Detection objects (same as /predict-single) |
| `total_processed` | int | Number of images successfully processed |
| `processing_time_seconds` | float | Total processing time |

#### Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid ZIP file"
}
```

**413 Payload Too Large:**
```json
{
  "detail": "ZIP file exceeds maximum size (50 MB)"
}
```

#### cURL Example

```bash
# Create ZIP file
zip batch_images.zip whale_*.jpg

# Upload batch
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/zip" \
  --data-binary "@batch_images.zip"
```

#### Python Example

```python
import requests
import zipfile
import os

# Create ZIP file
zip_path = "batch_images.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for filename in os.listdir("whale_images/"):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            zipf.write(f"whale_images/{filename}", filename)

# Upload batch
url = "http://localhost:8000/predict-batch"

with open(zip_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    print(f"Processed {data['total_processed']} images in {data['processing_time_seconds']:.2f}s")

    for result in data['results']:
        print(f"\n{result['image_ind']}:")
        print(f"  Species: {result['id_animal']}")
        print(f"  Confidence: {result['probability']:.2%}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

#### JavaScript Example

```javascript
// Create FormData with ZIP file
const formData = new FormData();
formData.append('file', zipFileInput.files[0]);

fetch('http://localhost:8000/predict-batch', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log(`Processed ${data.total_processed} images`);

    data.results.forEach(result => {
      console.log(`${result.image_ind}: ${result.id_animal} (${result.probability})`);
    });
  })
  .catch(error => console.error('Error:', error));
```

---

## Response Models

### Detection

Represents a single whale detection result.

```python
class Detection(BaseModel):
    image_ind: str              # Filename (e.g., "whale_001.jpg")
    bbox: list[int]            # [x, y, width, height]
    class_animal: str          # Individual ID (hex-like)
    id_animal: str             # Species name
    probability: float         # 0.0-1.0 confidence
    mask: str | None          # Base64 PNG (optional)
```

**Species Mapping:**

The `id_animal` field maps to one of the following species:
- Humpback Whale (Megaptera novaeangliae)
- Blue Whale (Balaenoptera musculus)
- Fin Whale (Balaenoptera physalus)
- Gray Whale (Eschrichtius robustus)
- Beluga Whale (Delphinapterus leucas)
- ... (15,587 total individuals across multiple species)

**Individual ID Format:**

The `class_animal` field contains a unique identifier:
- Format: Hex-like string (e.g., `a1b2c3d4e5f6`)
- Total individuals: 15,587
- Lookup: Via `whales_be_service/config.yaml`

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Check file format, ZIP structure |
| 413 | Payload Too Large | Reduce file size |
| 422 | Unprocessable Entity | Verify required parameters |
| 500 | Internal Server Error | Contact support, check logs |

### Error Response Format

```json
{
  "detail": "Human-readable error message"
}
```

### Common Errors

#### Unsupported Media Type

**Error:**
```json
{
  "detail": "Unsupported media type. Allowed: image/jpeg, image/png"
}
```

**Solution:**
- Convert image to JPG or PNG
- Check file extension matches content type

#### Invalid ZIP File

**Error:**
```json
{
  "detail": "Invalid ZIP file"
}
```

**Solution:**
- Verify ZIP is not corrupted
- Use standard ZIP format (not RAR, 7z)
- Check ZIP contains only images

#### No File Provided

**Error:**
```json
{
  "detail": "No file provided"
}
```

**Solution:**
- Ensure `file` field is present in form data
- Verify file is not empty

---

## Rate Limiting

**Current Status:** Not implemented

**Future Implementation:**
- Limit: 100 requests/minute per IP
- Headers:
  - `X-RateLimit-Limit`: Total allowed requests
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Unix timestamp for reset

**Exceeded Rate Limit Response (429):**
```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds."
}
```

---

## Code Examples

### Complete Python Client

```python
import requests
import base64
import zipfile
import os
from typing import Optional

class WhaleIdentificationClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def predict_single(self, image_path: str) -> dict:
        """Predict species for a single image."""
        url = f"{self.base_url}/predict-single"

        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        response.raise_for_status()
        return response.json()

    def predict_batch(self, image_dir: str, zip_path: Optional[str] = None) -> dict:
        """Predict species for multiple images."""
        # Create ZIP
        if zip_path is None:
            zip_path = "temp_batch.zip"

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filename in os.listdir(image_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(image_dir, filename)
                    zipf.write(file_path, filename)

        # Upload
        url = f"{self.base_url}/predict-batch"
        with open(zip_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        response.raise_for_status()
        return response.json()

    def save_mask(self, result: dict, output_path: str):
        """Save base64 mask to PNG file."""
        if result.get('mask'):
            mask_data = base64.b64decode(result['mask'])
            with open(output_path, "wb") as f:
                f.write(mask_data)

# Usage
client = WhaleIdentificationClient()

# Single prediction
result = client.predict_single("whale.jpg")
print(f"Species: {result['id_animal']}")
client.save_mask(result, "whale_mask.png")

# Batch prediction
batch_result = client.predict_batch("whale_images/")
print(f"Processed {batch_result['total_processed']} images")
```

### Node.js Client

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class WhaleIdentificationClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async predictSingle(imagePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));

    const response = await axios.post(
      `${this.baseUrl}/predict-single`,
      form,
      { headers: form.getHeaders() }
    );

    return response.data;
  }

  async predictBatch(zipPath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(zipPath));

    const response = await axios.post(
      `${this.baseUrl}/predict-batch`,
      form,
      { headers: form.getHeaders() }
    );

    return response.data;
  }

  saveMask(result, outputPath) {
    if (result.mask) {
      const buffer = Buffer.from(result.mask, 'base64');
      fs.writeFileSync(outputPath, buffer);
    }
  }
}

// Usage
const client = new WhaleIdentificationClient();

(async () => {
  // Single prediction
  const result = await client.predictSingle('whale.jpg');
  console.log(`Species: ${result.id_animal}`);
  client.saveMask(result, 'whale_mask.png');
})();
```

---

## Interactive API Documentation

The API provides interactive documentation via **Swagger UI**:

**URL:** http://localhost:8000/docs

**Features:**
- Try out endpoints directly in browser
- View request/response schemas
- Download OpenAPI specification

---

## Changelog

### Version 0.1.0 (Current)

**Endpoints:**
- ✅ POST /predict-single
- ✅ POST /predict-batch

**Features:**
- Background removal with rembg
- Vision Transformer inference
- Batch processing

**Planned (v0.2.0):**
- Rate limiting
- API authentication
- Webhook notifications
- Async processing for large batches

---

## Support

- **GitHub Issues:** [Report API issues](https://github.com/0x0000dead/whales-identification/issues)
- **Email:** konstantin.baltsat@example.com

---

**Next Steps:**
- [Usage Guide](Usage) - Learn how to use the API effectively
- [Testing](Testing) - Run integration tests
- [Architecture](Architecture) - Understand the backend implementation
