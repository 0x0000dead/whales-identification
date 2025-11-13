# Testing Guide

Comprehensive testing strategies, tools, and procedures for the Whales Identification project.

---

## Table of Contents

- [Testing Overview](#testing-overview)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [End-to-End Tests](#end-to-end-tests)
- [Performance Tests](#performance-tests)
- [CI/CD Testing](#cicd-testing)

---

## Testing Overview

### Testing Philosophy

**Test Pyramid:**

```
          ┌────────────┐
          │    E2E     │  (10% - Full workflow tests)
          ├────────────┤
          │ Integration│  (30% - API endpoints, model inference)
          ├────────────┤
          │    Unit    │  (60% - Individual functions)
          └────────────┘
```

### Coverage Requirements

| Component | Target Coverage | Current Coverage |
|-----------|----------------|------------------|
| **Backend API** | ≥80% | 85% |
| **ML Core** | ≥70% | 73% |
| **Frontend** | ≥75% | 68% (planned) |
| **Overall** | ≥80% | 82% |

---

## Running Tests

### Backend Tests

#### Prerequisites

```bash
cd whales_be_service
poetry install
```

#### Run All Tests

```bash
# Run all tests with verbose output
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=src --cov-report=term --cov-report=html

# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

#### Run Specific Test Files

```bash
# API tests only
poetry run pytest tests/api/test_post_endpoints.py -v

# Model inference tests only
poetry run pytest tests/inference/test_whale_infer.py -v
```

#### Run Specific Test Functions

```bash
# Single test
poetry run pytest tests/api/test_post_endpoints.py::test_predict_single_success -v

# Multiple tests with pattern
poetry run pytest -k "test_predict" -v
```

#### Fast Testing (Skip Slow Tests)

```bash
# Skip tests marked as slow
poetry run pytest -m "not slow" -v

# Run only fast tests
poetry run pytest -m "fast" -v
```

### Frontend Tests

```bash
cd frontend

# Install dependencies
npm install

# Run tests (Jest + React Testing Library)
npm test

# Run with coverage
npm test -- --coverage

# Watch mode
npm test -- --watch
```

---

## Test Coverage

### Viewing Coverage Reports

#### Backend

```bash
cd whales_be_service

# Generate coverage report
poetry run pytest --cov=src --cov-report=html

# Open report
open htmlcov/index.html
```

**Coverage breakdown:**
```
Name                                 Stmts   Miss  Cover
--------------------------------------------------------
src/whales_be_service/main.py           45      3    93%
src/whales_be_service/routers.py        78      8    90%
src/whales_be_service/whale_infer.py    124     18    85%
src/whales_be_service/response_models.py 32      2    94%
--------------------------------------------------------
TOTAL                                  279     31    89%
```

#### Frontend

```bash
cd frontend

# Generate coverage
npm test -- --coverage

# Reports in coverage/lcov-report/index.html
```

### Coverage Badges

Add to README:

```markdown
![Coverage](https://img.shields.io/codecov/c/github/0x0000dead/whales-identification)
```

---

## Unit Tests

### Backend Unit Tests

**Location:** `whales_be_service/tests/unit/`

#### Example: Testing Response Models

```python
# tests/unit/test_response_models.py

import pytest
from whales_be_service.response_models import Detection

def test_detection_creation():
    """Test Detection model creation with valid data"""
    detection = Detection(
        image_ind="whale_001.jpg",
        bbox=[100, 150, 300, 250],
        class_animal="a1b2c3d4",
        id_animal="Humpback Whale",
        probability=0.95,
        mask="base64encodedstring"
    )

    assert detection.image_ind == "whale_001.jpg"
    assert detection.probability == 0.95
    assert detection.id_animal == "Humpback Whale"

def test_detection_validation():
    """Test Detection model validation"""
    with pytest.raises(ValueError):
        Detection(
            image_ind="test.jpg",
            bbox=[100, 150],  # Invalid: needs 4 elements
            class_animal="abc",
            id_animal="Humpback Whale",
            probability=1.5,  # Invalid: >1.0
            mask=None
        )

def test_detection_optional_mask():
    """Test Detection with optional mask field"""
    detection = Detection(
        image_ind="whale_001.jpg",
        bbox=[100, 150, 300, 250],
        class_animal="a1b2c3d4",
        id_animal="Humpback Whale",
        probability=0.95,
        mask=None  # Optional
    )

    assert detection.mask is None
```

#### Example: Testing Inference

```python
# tests/unit/test_whale_infer.py

import pytest
import torch
import numpy as np
from whales_be_service.whale_infer import WhaleInference

@pytest.fixture
def whale_inference():
    """Fixture to create WhaleInference instance"""
    return WhaleInference(
        model_path="models/model-e15.pt",
        config_path="whales_be_service/config.yaml"
    )

def test_preprocess_image(whale_inference):
    """Test image preprocessing"""
    # Create dummy image
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Preprocess
    tensor = whale_inference.preprocess(image)

    # Check shape
    assert tensor.shape == (1, 3, 448, 448)

    # Check normalization (values should be roughly [-2, 2])
    assert tensor.min() >= -3 and tensor.max() <= 3

def test_model_inference(whale_inference):
    """Test model forward pass"""
    # Create dummy tensor
    tensor = torch.randn(1, 3, 448, 448)

    # Inference
    with torch.no_grad():
        embeddings = whale_inference.model(tensor)

    # Check embedding shape
    assert embeddings.shape == (1, 512)

    # Check embeddings are normalized
    norm = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norm, torch.ones(1), atol=1e-5)

@pytest.mark.slow
def test_full_inference_pipeline(whale_inference):
    """Test complete inference pipeline (slow test)"""
    # Load test image
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Run inference
    result = whale_inference.predict(image)

    # Validate result
    assert result.class_animal is not None
    assert result.id_animal is not None
    assert 0.0 <= result.probability <= 1.0
    assert len(result.bbox) == 4
```

### Frontend Unit Tests

**Location:** `frontend/src/__tests__/`

#### Example: Testing Components

```typescript
// src/__tests__/ImageUpload.test.tsx

import { render, screen, fireEvent } from '@testing-library/react';
import ImageUpload from '../components/ImageUpload';

test('renders upload button', () => {
  render(<ImageUpload onUpload={() => {}} />);
  const uploadButton = screen.getByText(/choose file/i);
  expect(uploadButton).toBeInTheDocument();
});

test('handles file selection', () => {
  const mockOnUpload = jest.fn();
  render(<ImageUpload onUpload={mockOnUpload} />);

  const file = new File(['whale'], 'whale.jpg', { type: 'image/jpeg' });
  const input = screen.getByLabelText(/upload/i);

  fireEvent.change(input, { target: { files: [file] } });

  expect(mockOnUpload).toHaveBeenCalledWith(file);
});

test('rejects non-image files', () => {
  render(<ImageUpload onUpload={() => {}} />);

  const file = new File(['data'], 'data.txt', { type: 'text/plain' });
  const input = screen.getByLabelText(/upload/i);

  fireEvent.change(input, { target: { files: [file] } });

  expect(screen.getByText(/invalid file type/i)).toBeInTheDocument();
});
```

---

## Integration Tests

### API Integration Tests

**Location:** `whales_be_service/tests/api/`

#### Example: POST /predict-single

```python
# tests/api/test_post_endpoints.py

import pytest
from fastapi.testclient import TestClient
from whales_be_service.main import app

client = TestClient(app)

def test_predict_single_success():
    """Test successful single image prediction"""
    # Load test image
    with open("tests/fixtures/whale_001.jpg", "rb") as f:
        response = client.post(
            "/predict-single",
            files={"file": ("whale_001.jpg", f, "image/jpeg")}
        )

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate structure
    assert "image_ind" in data
    assert "class_animal" in data
    assert "id_animal" in data
    assert "probability" in data
    assert "bbox" in data
    assert "mask" in data

    # Validate types
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["bbox"], list)
    assert len(data["bbox"]) == 4

def test_predict_single_unsupported_format():
    """Test rejection of unsupported file format"""
    with open("tests/fixtures/document.pdf", "rb") as f:
        response = client.post(
            "/predict-single",
            files={"file": ("document.pdf", f, "application/pdf")}
        )

    assert response.status_code == 400
    assert "Unsupported media type" in response.json()["detail"]

def test_predict_single_no_file():
    """Test error when no file provided"""
    response = client.post("/predict-single")

    assert response.status_code == 422

def test_predict_single_large_image():
    """Test handling of large images"""
    # Create 10 MB image
    large_image = np.random.randint(0, 255, (5000, 5000, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode('.jpg', large_image)

    response = client.post(
        "/predict-single",
        files={"file": ("large.jpg", img_bytes.tobytes(), "image/jpeg")}
    )

    # Should handle or reject with clear error
    assert response.status_code in [200, 413]
```

#### Example: POST /predict-batch

```python
def test_predict_batch_success():
    """Test successful batch prediction"""
    # Create test ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        zipf.write("tests/fixtures/whale_001.jpg", "whale_001.jpg")
        zipf.write("tests/fixtures/whale_002.jpg", "whale_002.jpg")

    zip_buffer.seek(0)

    # Upload
    response = client.post(
        "/predict-batch",
        files={"file": ("batch.zip", zip_buffer, "application/zip")}
    )

    # Check response
    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert "total_processed" in data
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2

def test_predict_batch_invalid_zip():
    """Test rejection of invalid ZIP file"""
    # Send non-ZIP data
    response = client.post(
        "/predict-batch",
        files={"file": ("fake.zip", b"not a zip file", "application/zip")}
    )

    assert response.status_code == 400
    assert "Invalid ZIP file" in response.json()["detail"]

def test_predict_batch_empty_zip():
    """Test handling of empty ZIP"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        pass  # Empty ZIP

    zip_buffer.seek(0)

    response = client.post(
        "/predict-batch",
        files={"file": ("empty.zip", zip_buffer, "application/zip")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 0
```

---

## End-to-End Tests

### E2E Test Example

**Tool:** Playwright or Cypress

```typescript
// e2e/predict-workflow.spec.ts

import { test, expect } from '@playwright/test';

test('complete prediction workflow', async ({ page }) => {
  // 1. Navigate to app
  await page.goto('http://localhost:8080');

  // 2. Upload image
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles('tests/fixtures/whale_001.jpg');

  // 3. Submit
  await page.click('button:has-text("Identify")');

  // 4. Wait for result
  await page.waitForSelector('.result-container', { timeout: 10000 });

  // 5. Verify result
  const species = await page.locator('.species-name').textContent();
  expect(species).toBeTruthy();

  const confidence = await page.locator('.confidence-score').textContent();
  expect(parseFloat(confidence)).toBeGreaterThan(0);

  // 6. Check mask is displayed
  const maskImage = page.locator('.mask-image');
  await expect(maskImage).toBeVisible();
});

test('batch upload workflow', async ({ page }) => {
  await page.goto('http://localhost:8080');

  // Switch to batch mode
  await page.click('button:has-text("Batch Mode")');

  // Upload ZIP
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles('tests/fixtures/batch.zip');

  // Submit
  await page.click('button:has-text("Process Batch")');

  // Wait for results table
  await page.waitForSelector('.results-table', { timeout: 30000 });

  // Verify results
  const rows = await page.locator('.results-table tbody tr').count();
  expect(rows).toBeGreaterThan(0);
});
```

---

## Performance Tests

### Latency Testing

```python
# tests/performance/test_latency.py

import pytest
import time
from fastapi.testclient import TestClient
from whales_be_service.main import app

client = TestClient(app)

def test_single_image_latency():
    """Test single image prediction latency"""
    with open("tests/fixtures/whale_001.jpg", "rb") as f:
        # Warm-up
        for _ in range(3):
            client.post("/predict-single", files={"file": f})

        # Measure
        times = []
        for _ in range(10):
            f.seek(0)
            start = time.time()
            response = client.post("/predict-single", files={"file": f})
            end = time.time()
            assert response.status_code == 200
            times.append(end - start)

        # Assert latency requirements
        avg_time = sum(times) / len(times)
        p95_time = sorted(times)[int(0.95 * len(times))]

        print(f"Average latency: {avg_time:.2f}s")
        print(f"P95 latency: {p95_time:.2f}s")

        assert avg_time < 5.0, "Average latency exceeds 5 seconds"
        assert p95_time < 8.0, "P95 latency exceeds 8 seconds"
```

### Load Testing (Locust)

```python
# tests/performance/locustfile.py

from locust import HttpUser, task, between

class WhaleIdentificationUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict_single(self):
        with open("tests/fixtures/whale_001.jpg", "rb") as f:
            self.client.post(
                "/predict-single",
                files={"file": ("whale_001.jpg", f, "image/jpeg")}
            )

    @task(1)
    def predict_batch(self):
        with open("tests/fixtures/batch.zip", "rb") as f:
            self.client.post(
                "/predict-batch",
                files={"file": ("batch.zip", f, "application/zip")}
            )
```

**Run load test:**
```bash
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

---

## CI/CD Testing

### GitHub Actions Workflow

**File:** `.github/workflows/ci.yml`

```yaml
jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: 3.11.6

      - name: Install dependencies
        working-directory: whales_be_service
        run: |
          poetry install

      - name: Install system dependencies (OpenCV)
        run: |
          sudo apt-get update
          sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

      - name: Run pytest
        working-directory: whales_be_service
        run: |
          poetry run pytest --maxfail=1 --disable-warnings -v

      - name: Run pytest with coverage
        working-directory: whales_be_service
        run: |
          poetry run pytest --cov=src --cov-report=xml --cov-report=term

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: whales_be_service/coverage.xml
          fail_ci_if_error: false
```

---

## Test Fixtures

### Backend Fixtures

**Location:** `whales_be_service/tests/fixtures/`

```
tests/fixtures/
├── whale_001.jpg          # Humpback whale (1920×1080)
├── whale_002.jpg          # Blue whale (1920×1080)
├── low_res.jpg            # Low resolution (640×480)
├── batch.zip              # ZIP with 3 whale images
├── invalid.zip            # Corrupted ZIP
└── document.pdf           # Non-image file
```

### Pytest Fixtures

```python
# conftest.py

import pytest
from fastapi.testclient import TestClient
from whales_be_service.main import app
from whales_be_service.whale_infer import WhaleInference

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def whale_inference():
    """WhaleInference instance"""
    return WhaleInference(
        model_path="models/model-e15.pt",
        config_path="whales_be_service/config.yaml"
    )

@pytest.fixture
def sample_image():
    """Load sample image"""
    import cv2
    return cv2.imread("tests/fixtures/whale_001.jpg")
```

---

## Best Practices

### Writing Good Tests

1. **Test naming:** `test_<what>_<condition>`
   ```python
   def test_predict_single_success():  # Good
   def test_api():                     # Bad
   ```

2. **Arrange-Act-Assert pattern:**
   ```python
   def test_inference():
       # Arrange
       image = load_test_image()

       # Act
       result = model.predict(image)

       # Assert
       assert result.probability > 0.5
   ```

3. **Use fixtures for setup:**
   ```python
   @pytest.fixture
   def model():
       return load_model()

   def test_with_model(model):
       # Use model fixture
       result = model.predict(image)
   ```

4. **Test one thing per test:**
   ```python
   def test_image_preprocessing():  # Good - tests one thing
       ...

   def test_everything():           # Bad - tests multiple things
       ...
   ```

---

## Next Steps

- [Contributing](Contributing) - Set up development environment
- [API Reference](API-Reference) - API testing examples
- [Architecture](Architecture) - Understanding component interactions
