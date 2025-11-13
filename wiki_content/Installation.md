# Installation Guide

This guide provides step-by-step instructions for installing and running the Whales Identification project.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [Method 1: Docker Compose (Recommended)](#method-1-docker-compose-recommended)
  - [Method 2: Local Development](#method-2-local-development)
  - [Method 3: Streamlit Demo Only](#method-3-streamlit-demo-only)
- [Downloading Models](#downloading-models)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.11.6 | Backend runtime |
| **Node.js** | ≥16.0 | Frontend build |
| **Docker** | ≥20.10 | Containerization |
| **Docker Compose** | ≥2.0 | Multi-container orchestration |
| **Git** | Any | Version control |
| **Poetry** | ≥1.5 | Python package manager |

### System Requirements

- **RAM:** Minimum 8GB (16GB recommended for training)
- **Storage:** ~5GB for models + dependencies
- **GPU:** Optional (CUDA-compatible for faster inference)

---

## Installation Methods

### Method 1: Docker Compose (Recommended)

This method provides a complete stack (Frontend + Backend + API Docs) with minimal setup.

#### Step 1: Clone Repository

```bash
git clone https://github.com/0x0000dead/whales-identification.git
cd whales-identification
```

#### Step 2: Install Hugging Face CLI

```bash
pip install huggingface_hub
```

#### Step 3: Download Models

```bash
# Make script executable (if needed)
chmod +x scripts/download_models.sh

# Download models from Hugging Face
./scripts/download_models.sh
```

**Expected output:**
```
Downloading model-e15.pt...
✓ Downloaded to models/model-e15.pt (2.1 GB)
```

**Alternative (manual download):**
- [Hugging Face](https://huggingface.co/baltsat/Whales-Identification/tree/main)
- [Yandex Disk](https://disk.yandex.ru/d/GshqU9o6nNz7ZA)

Place models in `models/` directory.

#### Step 4: Start Services

```bash
docker compose up --build
```

**First build may take 10-15 minutes** (downloads dependencies, builds images).

#### Step 5: Verify Services

Open in browser:
- **Backend API:** http://localhost:8000/docs (Swagger UI)
- **Frontend UI:** http://localhost:8080
- **Health Check:** http://localhost:8000/docs

**Expected:**
- Swagger UI shows 2 endpoints: `/predict-single`, `/predict-batch`
- Frontend displays file upload interface

---

### Method 2: Local Development

For development with hot-reload and debugging.

#### Backend Setup

```bash
# Navigate to backend directory
cd whales_be_service

# Install Poetry (if not installed)
pip install poetry

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Install OpenCV system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# macOS
brew install opencv  # Usually not required on macOS

# Download models (from project root)
cd ..
./scripts/download_models.sh

# Start backend (from whales_be_service)
cd whales_be_service
poetry run python -m uvicorn whales_be_service.main:app \
  --host 0.0.0.0 --port 8000 --reload
```

**Backend will be available at:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

#### Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Frontend will be available at:**
- Dev Server: http://localhost:5173

**Production build:**
```bash
npm run build      # Build to frontend/dist
npm run preview    # Preview production build
```

---

### Method 3: Streamlit Demo Only

For quick demonstration without full API setup.

#### Step 1: Navigate to Demo Directory

```bash
cd research/demo-ui
```

#### Step 2: Install Dependencies

```bash
# Install Poetry (if not installed)
pip install poetry

# Install dependencies
poetry install
```

#### Step 3: Download Models

```bash
# From project root
cd ../..
./scripts/download_models.sh
cd research/demo-ui
```

#### Step 4: Run Streamlit App

```bash
poetry run streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

**App will be available at:**
- http://localhost:8501

**Alternative demo (with masking):**
```bash
cd ../demo-ui-mask
poetry install
poetry run streamlit run streamlit_app.py --server.port=8502
```

---

## Downloading Models

### Option 1: Automated Script (Recommended)

```bash
# From project root
./scripts/download_models.sh
```

**What it does:**
1. Creates `models/` directory
2. Uses `huggingface-cli` to download `model-e15.pt` (~2.1 GB)
3. Verifies download integrity

**Requirements:**
- `huggingface_hub` installed: `pip install huggingface_hub`

### Option 2: Manual Download

#### From Hugging Face

1. Visit [baltsat/Whales-Identification](https://huggingface.co/baltsat/Whales-Identification/tree/main)
2. Download `model-e15.pt` (2.1 GB)
3. Place in `models/` directory

#### From Yandex Disk

1. Visit [Yandex Disk link](https://disk.yandex.ru/d/GshqU9o6nNz7ZA)
2. Download all models
3. Place in `models/` directory

**Directory structure:**
```
whales-identification/
├── models/
│   └── model-e15.pt  (2.1 GB)
├── whales_be_service/
├── frontend/
└── research/
```

---

## Verification

### Test Backend API

#### Using curl

```bash
# Health check
curl http://localhost:8000/docs

# Single image prediction
curl -X POST "http://localhost:8000/predict-single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/whale_image.jpg"
```

#### Using Python

```python
import requests

# Single image
with open("whale_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict-single",
        files={"file": f}
    )
    print(response.json())
```

**Expected response:**
```json
{
  "image_ind": "whale_image.jpg",
  "bbox": [100, 150, 300, 250],
  "class_animal": "a1b2c3d4",
  "id_animal": "Humpback Whale",
  "probability": 0.95,
  "mask": "iVBORw0KGgoAAAANS..."
}
```

### Test Frontend

1. Open http://localhost:8080
2. Click "Upload Image"
3. Select a whale image
4. Verify results display correctly

### Run Tests

```bash
# From whales_be_service directory
cd whales_be_service

# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=term --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Expected output:**
```
tests/api/test_post_endpoints.py::test_predict_single_success PASSED
tests/api/test_post_endpoints.py::test_predict_batch_success PASSED
...
Coverage: 85%
```

---

## Troubleshooting

### Issue 1: `ImportError: libGL.so.1: cannot open shared object file`

**Cause:** Missing OpenCV system dependencies

**Solution (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

**Solution (macOS):**
```bash
brew install opencv  # Usually not required
```

---

### Issue 2: `huggingface-cli: command not found`

**Cause:** Hugging Face CLI not installed

**Solution:**
```bash
pip install huggingface_hub
```

**Verify:**
```bash
huggingface-cli --version
```

---

### Issue 3: `Poetry could not find a pyproject.toml file`

**Cause:** Running command from wrong directory

**Solution:**
```bash
# Backend commands must run from whales_be_service/
cd whales_be_service
poetry install

# Frontend commands must run from frontend/
cd ../frontend
npm install
```

---

### Issue 4: `docker: Error response from daemon: no such image`

**Cause:** Docker images not built

**Solution:**
```bash
# Build images
docker compose build

# Force rebuild (if needed)
docker compose build --no-cache
```

---

### Issue 5: Models not found error

**Cause:** Models not downloaded to `models/` directory

**Solution:**
```bash
# Check models directory
ls -lh models/

# If empty, download models
./scripts/download_models.sh

# Verify model exists
ls -lh models/model-e15.pt
```

---

### Issue 6: Port already in use

**Cause:** Another service using port 8000 or 8080

**Solution (macOS/Linux):**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Solution (Windows):**
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

### Issue 7: Docker permission denied (Linux)

**Cause:** User not in docker group

**Solution:**
```bash
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

---

### Issue 8: Pre-commit hooks failing

**Cause:** Code doesn't meet quality standards

**Solution:**
```bash
# Auto-fix formatting
poetry run black .
poetry run isort .

# Check linting
poetry run flake8 .

# Run all hooks manually
poetry run pre-commit run --all-files
```

---

## Next Steps

- **[Usage Guide](Usage)** - Learn how to use the API and frontend
- **[API Reference](API-Reference)** - Detailed API documentation
- **[Contributing](Contributing)** - Set up development environment
- **[FAQ](FAQ)** - More troubleshooting tips

---

**Need help?** [Open an issue](https://github.com/0x0000dead/whales-identification/issues) or [start a discussion](https://github.com/0x0000dead/whales-identification/discussions).
