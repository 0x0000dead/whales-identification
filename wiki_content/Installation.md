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

| Software           | Version | Purpose                       |
| ------------------ | ------- | ----------------------------- |
| **Python**         | 3.11.6  | Backend runtime               |
| **Node.js**        | ≥20.19  | Frontend build                |
| **Docker**         | ≥20.10  | Containerization              |
| **Docker Compose** | ≥2.0    | Multi-container orchestration |
| **Git**            | Any     | Version control               |
| **Poetry**         | ≥1.5    | Python package manager        |

### System Requirements

- **RAM:** Minimum 8GB (16GB recommended for training)
- **Storage:** ~5GB for models + dependencies
- **GPU:** Optional ([CUDA-compatible](#gpu-acceleration-optional) for faster inference — see the [NVIDIA Container Toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

### GPU Acceleration (Optional)

To use GPU acceleration with Docker containers, you need to install the NVIDIA Container Toolkit:

1. **Install NVIDIA drivers** for your GPU
2. **Install NVIDIA Container Toolkit** - Follow the official guide: [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Quick install (Ubuntu/Debian):**

```bash
# Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**Running the stack with GPU** — use the GPU overlay file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Verify the backend picked up the GPU: `curl http://localhost:8000/health` →
`"device": "cuda:0"`.

**Note:** GPU support is optional. The project works on CPU-only systems with slower inference times.

---

## Installation Methods

### Method 1: Docker Compose (Recommended)

This method provides a complete stack (Frontend + Backend + API Docs) with minimal setup.

#### Step 1: Clone Repository

```bash
git clone https://github.com/0x0000dead/whales-identification.git
cd whales-identification
```

#### Step 2: Start Services

```bash
docker compose up --build
```

> **No model download needed for Docker.** Model weights are baked into the
> backend image; on first boot `docker-entrypoint.sh` automatically downloads
> anything missing from Hugging Face, and named volumes keep the files
> between restarts. (Downloading models manually is only required for
> Method 2: Local Development.)

**First build may take 10-15 minutes** (downloads dependencies, builds images).

**Network access from other devices:** works out of the box. By default
`VITE_BACKEND` is empty — the frontend calls the backend at
`http://<host the UI is opened from>:8000` — and the dev compose sets
`ALLOWED_ORIGINS=*`. Just open `http://<machine-IP>:8080` from any device on
the network. Set `VITE_BACKEND` only for reverse-proxy setups or a
non-standard backend port.

**GPU mode (optional):** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, start the stack with the GPU overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Verify with `curl http://localhost:8000/health` — expect `"device": "cuda:0"`.

#### Step 3: Verify Services

Open in browser:

- **Backend API:** http://localhost:8000/docs (Swagger UI)
- **Frontend UI:** http://localhost:8080
- **Health Check:** http://localhost:8000/health

**Expected:**

- Swagger UI lists the API endpoints: `/v1/predict-single`, `/v1/predict-batch`, `/health`, `/metrics` (plus backwards-compatible aliases `/predict-single`, `/predict-batch`)
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

# macOS (only if you encounter import errors - usually not required):
brew install opencv
# Note: opencv-python package already includes system libraries on macOS

# Download models (from project root)
cd ..
pip install huggingface_hub==0.20.3   # 0.21+ moved huggingface-cli into an extra
./scripts/download_models.sh

# Start backend (from whales_be_service)
cd whales_be_service
poetry run python -m uvicorn whales_be_service.main:app \
  --host 0.0.0.0 --port 8000 --reload
```

**Expected output of `./scripts/download_models.sh`:**

```
→ 0x0000dead/ecomarineai-cetacean-effb4 / efficientnet_b4_512_fold0.ckpt → whales_be_service/src/whales_be_service/models/
✓ SHA256 OK
→ 0x0000dead/ecomarineai-cetacean-effb4 / encoder_classes.npy → whales_be_service/src/whales_be_service/models/
✓ SHA256 OK
→ 0x0000dead/ecomarineai-cetacean-effb4 / species_map.csv → whales_be_service/src/whales_be_service/resources/
✓ SHA256 OK
→ 0x0000dead/ecomarineai-cetacean-effb4 / anti_fraud_threshold.yaml → whales_be_service/src/whales_be_service/configs/
✓ SHA256 OK
→ 0x0000dead/ecomarineai-cetacean-effb4 / metrics_baseline.json → reports/
✓ SHA256 OK
→ baltsat/Whales-Identification / resnet101.pth → whales_be_service/src/whales_be_service/models/   (legacy)
✓ SHA256 OK
```

> **Note:** on the first backend start, `open_clip` additionally downloads the
> CLIP ViT-B-32 weights (~605 MB, `open_clip_model.safetensors` from
> `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) for the anti-fraud gate. In the
> Docker image these weights are already baked in.

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

**Environment Variables:**

| Variable       | Default                          | Description                       |
| -------------- | -------------------------------- | --------------------------------- |
| `VITE_BACKEND` | empty (runtime same-host:8000)   | Backend API URL override (build-time) |

**Network Access Configuration:**

By default `VITE_BACKEND` is empty and the frontend resolves the backend at
runtime as `http://<host the page is opened from>:8000` — opening the UI from
another machine works without configuration. Set `VITE_BACKEND` only when the
API lives behind a reverse proxy or on a non-standard port:

```bash
VITE_BACKEND=https://api.example.com npm run dev
```

When serving the UI to other machines, also start the backend with
`ALLOWED_ORIGINS=*` (or an explicit origin list) — the dev Docker Compose
already does this.

**Production build:**

```bash
npm run build      # Build to frontend/dist
npm run preview    # Preview production build
```

**Production build with custom backend URL:**

```bash
VITE_BACKEND=http://your-server-ip:8000 npm run build
```

---

### Method 3: Streamlit Demo Only

For quick demonstration without full API setup.

#### Step 1: Navigate to Demo Directory

```bash
cd research/demo-ui
```

#### Step 2: Install Dependencies

The demo uses Poetry for dependency management with its own `pyproject.toml`.

```bash
# Install Poetry (if not installed)
pip install poetry

# Install dependencies from pyproject.toml
poetry install
```

**Note:** This installs Streamlit, PyTorch, OpenCV, Albumentations, and other required packages.

#### Step 3: Download Models

The Streamlit demo uses the **legacy Vision Transformer model** (`model-e15.pt`), which is separate from the production EfficientNet-B4 model.

```bash
# The production download script downloads efficientnet_b4_512_fold0.ckpt
# For the Streamlit demo, download model-e15.pt manually:
mkdir -p research/demo-ui/models/
```

**Manual download (required for Streamlit demo):** Download `model-e15.pt` from [Yandex Disk](https://disk.yandex.com/d/lH17kkrYgv2-1w) and place in `research/demo-ui/models/`.

> **Note:** The production FastAPI backend uses `efficientnet_b4_512_fold0.ckpt` (EfficientNet-B4 ArcFace), downloaded via `./scripts/download_models.sh`. The Streamlit demo is a legacy prototype demonstrating the earlier ViT architecture.

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

1. Creates the target directories
2. Uses `huggingface-cli` to download 6 files:
   - `efficientnet_b4_512_fold0.ckpt` (production model) → `whales_be_service/src/whales_be_service/models/`
   - `encoder_classes.npy` → `whales_be_service/src/whales_be_service/models/`
   - `species_map.csv` → `whales_be_service/src/whales_be_service/resources/`
   - `anti_fraud_threshold.yaml` → `whales_be_service/src/whales_be_service/configs/`
   - `metrics_baseline.json` → `reports/`
   - `resnet101.pth` (legacy) → `whales_be_service/src/whales_be_service/models/`
3. Verifies each file against `models/checksums.sha256` (`✓ SHA256 OK`)

**Requirements:**

- `huggingface_hub` installed: `pip install huggingface_hub==0.20.3`
  (the script installs it automatically when `huggingface-cli` is missing)

### Option 2: Manual Download

#### From Hugging Face

1. Visit [0x0000dead/ecomarineai-cetacean-effb4](https://huggingface.co/0x0000dead/ecomarineai-cetacean-effb4/tree/main) (production) and [baltsat/Whales-Identification](https://huggingface.co/baltsat/Whales-Identification/tree/main) (legacy)
2. Download the files listed above
3. Place them into the directories listed above

#### From Yandex Disk

1. Visit [Yandex Disk link](https://disk.yandex.ru/d/GshqU9o6nNz7ZA)
2. Download all models
3. Place them into the directories listed above

**Directory structure:**

```
whales-identification/
├── whales_be_service/
│   └── src/whales_be_service/
│       ├── models/
│       │   ├── efficientnet_b4_512_fold0.ckpt
│       │   ├── encoder_classes.npy
│       │   └── resnet101.pth        # legacy
│       ├── resources/
│       │   └── species_map.csv
│       └── configs/
│           └── anti_fraud_threshold.yaml
├── reports/
│   └── metrics_baseline.json
├── frontend/
└── research/
```

---

## Verification

### Test Backend API

#### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Single image prediction
curl -X POST "http://localhost:8000/v1/predict-single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/whale_image.jpg"
```

#### Using Python

```python
import requests

# Single image
with open("whale_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/predict-single",
        files={"file": f}
    )
    print(response.json())
```

**Expected response:**

```json
{
  "image_ind": "whale_image.jpg",
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
Usually not required on macOS - the `opencv-python` package already includes all required system libraries. Only install if the above packages don't resolve the issue:

```bash
brew install opencv
```

---

### Issue 2: `huggingface-cli: command not found`

**Cause:** Hugging Face CLI not installed

**Solution:**

```bash
pip install huggingface_hub==0.20.3
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

**Cause:** Models not downloaded (local development without Docker). In Docker
the weights are baked into the image and re-downloaded automatically by the
entrypoint, so this issue applies to Method 2 only.

**Solution:**

```bash
# Check the service models directory
ls -lh whales_be_service/src/whales_be_service/models/

# If empty, download models (from project root)
./scripts/download_models.sh

# Verify the production checkpoint exists
ls -lh whales_be_service/src/whales_be_service/models/efficientnet_b4_512_fold0.ckpt
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
