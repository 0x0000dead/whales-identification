# Frequently Asked Questions (FAQ)

Common questions and solutions for the Whales Identification project.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Issues](#model-issues)
- [API Issues](#api-issues)
- [Docker Issues](#docker-issues)
- [Performance Issues](#performance-issues)
- [Licensing Questions](#licensing-questions)
- [Usage Questions](#usage-questions)

---

## Installation Issues

### Q: `ImportError: libGL.so.1: cannot open shared object file`

**Problem:** Missing OpenCV system dependencies

**Solution (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

**Solution (macOS):**
```bash
brew install opencv  # Usually not required on macOS
```

**Why it happens:** OpenCV (cv2) requires system libraries for image processing. These are not installed by default on some Linux distributions.

---

### Q: `huggingface-cli: command not found`

**Problem:** Hugging Face CLI not installed

**Solution:**
```bash
pip install huggingface_hub
```

**Verify:**
```bash
huggingface-cli --version
```

**Why it happens:** The model download script uses `huggingface-cli` to download models from Hugging Face Hub.

---

### Q: `Poetry could not find a pyproject.toml file`

**Problem:** Running command from wrong directory

**Solution:**
```bash
# Backend commands must run from whales_be_service/
cd whales_be_service
poetry install

# Frontend commands must run from frontend/
cd frontend
npm install
```

**Why it happens:** Poetry looks for `pyproject.toml` in the current directory. The backend's `pyproject.toml` is in `whales_be_service/`, not the project root.

---

### Q: `docker: Error response from daemon: no such image`

**Problem:** Docker images not built

**Solution:**
```bash
# Build images
docker compose build

# Force rebuild (if needed)
docker compose build --no-cache

# Start services
docker compose up
```

**Why it happens:** Docker needs to build images before running containers.

---

## Model Issues

### Q: Where do I download the models?

**Answer:** Models are available from two sources:

1. **Hugging Face (Recommended):**
   ```bash
   ./scripts/download_models.sh
   ```
   - URL: https://huggingface.co/baltsat/Whales-Identification/tree/main
   - File: `model-e15.pt` (2.1 GB)

2. **Yandex Disk (Alternative):**
   - URL: https://disk.yandex.ru/d/GshqU9o6nNz7ZA
   - Download all models

**Verify download:**
```bash
ls -lh models/model-e15.pt
# Should show ~2.1 GB file
```

---

### Q: Models not found error when starting API

**Problem:** Models not downloaded or wrong path

**Solution:**
```bash
# 1. Check models directory exists
ls models/

# 2. Download models
./scripts/download_models.sh

# 3. Verify model exists
ls -lh models/model-e15.pt

# 4. Check path in whale_infer.py
# Should be: models/model-e15.pt
```

**Why it happens:** The API expects models in the `models/` directory, but they are not committed to git (`.gitignore`).

---

### Q: Model inference is very slow (>30 seconds)

**Problem:** Running on CPU instead of GPU

**Solution:**
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Expected times:**
- GPU (V100): 2-4 seconds
- CPU: 5-15 seconds
- If >30 seconds: check for bottlenecks

---

### Q: Out of memory error during inference

**Problem:** Image too large or batch size too big

**Solution:**
```python
# Resize large images before inference
from PIL import Image

def resize_if_needed(image_path, max_size=1920):
    img = Image.open(image_path)
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)
    return img
```

**Or reduce batch size:**
```python
# Instead of batch_size=32
batch_size = 16  # Or even 8
```

---

## API Issues

### Q: API returns 400: Unsupported media type

**Problem:** File format not supported or MIME type mismatch

**Solution:**
```bash
# Supported formats: JPG, PNG, JPEG
# Convert if needed
convert whale.bmp whale.jpg

# Verify MIME type
file --mime-type whale.jpg
# Should be: image/jpeg or image/png
```

**Why it happens:** API only accepts `image/jpeg` and `image/png` content types for security.

---

### Q: API returns 500: Internal server error

**Problem:** Multiple possible causes

**Solution:**
```bash
# 1. Check logs
docker compose logs backend

# 2. Common causes:
# - Model not loaded: download models
# - OpenCV error: install system dependencies
# - CUDA error: check GPU availability

# 3. Test with simple request
curl -X POST "http://localhost:8000/predict-single" \
  -F "file=@small_test_image.jpg"
```

---

### Q: Frontend can't connect to backend

**Problem:** CORS error or wrong backend URL

**Solution:**
```bash
# 1. Verify backend is running
curl http://localhost:8000/docs

# 2. Check CORS settings in main.py
# Should allow frontend origin:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    ...
)

# 3. Check frontend API URL
# In frontend/.env or vite.config.ts
VITE_BACKEND_URL=http://localhost:8000
```

---

## Docker Issues

### Q: Port already in use

**Problem:** Another service using port 8000 or 8080

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

**Alternative:** Change ports in `docker-compose.yml`:
```yaml
services:
  backend:
    ports:
      - "8001:8000"  # Change 8000 → 8001
```

---

### Q: Docker permission denied (Linux)

**Problem:** User not in docker group

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Refresh groups
newgrp docker

# Verify
docker ps
```

**Why it happens:** Docker daemon requires root or docker group membership.

---

### Q: Docker build fails with "no space left on device"

**Problem:** Docker out of disk space

**Solution:**
```bash
# Clean up old images
docker system prune -a

# Check space
df -h

# Remove unused volumes
docker volume prune
```

---

## Performance Issues

### Q: Batch processing is taking too long

**Problem:** Processing 100 images sequentially

**Solution:**
```python
# Use multiprocessing for batch inference
from concurrent.futures import ThreadPoolExecutor

def process_batch_parallel(images, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(model.predict, images))
    return results
```

**Or use GPU batch processing:**
```python
# Process in batches of 16
batch_size = 16
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    tensor_batch = torch.stack([preprocess(img) for img in batch])
    predictions = model(tensor_batch)
```

---

### Q: High memory usage

**Problem:** Model loaded multiple times or not released

**Solution:**
```python
# Ensure model is loaded once
if not hasattr(app.state, 'model'):
    app.state.model = load_model()

# Use context manager for inference
with torch.no_grad():
    predictions = model(tensor)

# Clear CUDA cache if using GPU
import torch
torch.cuda.empty_cache()
```

---

## Licensing Questions

### Q: Can I use this project commercially?

**Answer:** ⚠️ **No, commercial use is prohibited** due to:

1. **HappyWhale data:** CC-BY-NC-4.0 (non-commercial)
2. **Ministry RF data:** Research-only license
3. **ImageNet pretrained weights:** Non-commercial terms

**For commercial use:**
- Train models from scratch with your own data
- Use commercial datasets
- Contact data providers for commercial licenses

**See:**
- [LICENSE_MODELS.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_MODELS.md)
- [LICENSE_DATA.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_DATA.md)

---

### Q: Can I modify the code?

**Answer:** ✅ **Yes!** The code is MIT licensed.

**You can:**
- Modify the code
- Use in your own projects
- Fork the repository
- Contribute back via pull requests

**You must:**
- Include original copyright notice
- Include MIT license text

**See:** [LICENSE](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE)

---

### Q: Can I use the models for research?

**Answer:** ✅ **Yes!** Models can be used for non-commercial research.

**Requirements:**
- Cite the original datasets (HappyWhale, Ministry RF)
- Acknowledge the project in publications
- Share results with the community

**Citation:**
```bibtex
@software{whales_identification_2024,
  author = {Baltsat, Konstantin and Tarasov, Artem and Vandanov, Sergey and Serov, Alexandr},
  title = {Whales Identification: ML Library for Marine Mammal Detection},
  year = {2024},
  url = {https://github.com/0x0000dead/whales-identification}
}
```

---

## Usage Questions

### Q: What image quality is required?

**Answer:**

**Recommended:**
- Resolution: ≥1920×1080
- Format: JPG or PNG
- Lighting: Good natural lighting
- Angle: Aerial view, 30-45° angle
- Distance: Whale fills 20-80% of frame

**Minimum:**
- Resolution: ≥800×600
- No extreme blur or occlusion
- Whale clearly visible

**Accuracy impact:**
- High-quality: 90-93% precision
- Low-quality (<800×600): 70-80% precision (15-20% drop)

---

### Q: How many images can I process at once?

**Answer:**

**Limits:**
- Single image: Max 10 MB
- Batch ZIP: Max 50 MB, max 100 images

**Recommendations:**
- Small batches (1-10): <1 minute
- Medium batches (10-50): 1-5 minutes
- Large batches (50-100): 5-15 minutes

**For larger datasets:**
- Split into multiple ZIP files
- Use script to process in chunks:
```python
import os
import zipfile

def split_batch(image_dir, max_per_zip=50):
    images = sorted(os.listdir(image_dir))
    for i in range(0, len(images), max_per_zip):
        batch = images[i:i+max_per_zip]
        zip_name = f"batch_{i//max_per_zip + 1}.zip"
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            for img in batch:
                zipf.write(os.path.join(image_dir, img), img)
        print(f"Created {zip_name} with {len(batch)} images")
```

---

### Q: Which model should I use?

**Answer:** Depends on your use case:

**Best accuracy (93%):**
- Vision Transformer L/32
- Use for: Research, validation, high-value species

**Production API (91%, 2s):**
- Vision Transformer B/16
- Use for: API deployments, GPU servers

**Real-time (<1s, 88%):**
- EfficientNet-B0
- Use for: Real-time apps, edge devices

**Edge devices (82%, 0.8s):**
- ResNet-54
- Use for: Jetson Nano, low-power devices

**See [Model Cards](Model-Cards) for detailed comparison.**

---

### Q: Can it detect multiple whales in one image?

**Answer:** ⚠️ **Current version: No**

The model predicts **one whale per image**. For multiple whales:

**Workaround:**
1. Manually crop each whale
2. Upload each crop separately

**Planned feature (v0.2.0):**
- Object detection with YOLO/Faster R-CNN
- Automatic cropping of multiple whales
- Batch prediction on all detections

---

### Q: What species are supported?

**Answer:** 1,000 individual whales and dolphins across species including:

- Humpback Whale (Megaptera novaeangliae)
- Blue Whale (Balaenoptera musculus)
- Fin Whale (Balaenoptera physalus)
- Gray Whale (Eschrichtius robustus)
- Beluga Whale (Delphinapterus leucas)
- Right Whale (Eubalaena spp.)
- Sperm Whale (Physeter macrocephalus)
- Orca (Orcinus orca)
- Bottlenose Dolphin (Tursiops truncatus)
- Spinner Dolphin (Stenella longirostris)
- ... and more

**Full mapping:** `whales_be_service/config.yaml`

**Dataset:** ~80,000 images (~60,000 train + ~20,000 test) labeled for 1,000 individual marine mammals

---

### Q: How accurate is the identification?

**Answer:**

**Overall metrics (Vision Transformer L/32):**
- Precision@1: 93.2%
- Precision@5: 97.8%
- Recall (Sensitivity): 91.5%
- Specificity: 92.3%
- F1-Score: 0.923
- Inference Time: 3.5s (GPU), 7.5s (CPU)

**ТЗ Compliance:** ✅ All metrics exceed requirements (Precision ≥80%, Recall >85%, Specificity >90%, F1 >0.6, Time ≤8s)

**Per-species (top performers):**
- Humpback Whale: 95.3%
- Blue Whale: 94.1%
- Orca: 94.8%

**Limitations:**
- 15-20% accuracy drop on:
  - Low-resolution (<800×600)
  - Heavy occlusion (>50%)
  - Poor lighting (night, fog)
  - Extreme angles

**See [Model Cards](Model-Cards) for detailed metrics.**

---

## Still Need Help?

### Documentation

- [Installation Guide](Installation) - Setup instructions
- [Usage Guide](Usage) - How to use the API
- [API Reference](API-Reference) - Complete API docs
- [Architecture](Architecture) - System design
- [Testing](Testing) - Testing guide
- [Contributing](Contributing) - Development guide

### Support Channels

- **GitHub Issues:** [Report a bug](https://github.com/0x0000dead/whales-identification/issues)
- **GitHub Discussions:** [Ask a question](https://github.com/0x0000dead/whales-identification/discussions)

### Before Asking

1. ✅ Search existing issues
2. ✅ Check documentation
3. ✅ Review this FAQ
4. ✅ Try troubleshooting steps

### When Opening an Issue

**Provide:**
- ✅ Clear description of problem
- ✅ Steps to reproduce
- ✅ Error messages (full stack trace)
- ✅ Environment info (OS, Python version, Docker version)
- ✅ Expected vs actual behavior

**Example:**
```markdown
## Problem
API returns 500 error when uploading image

## Steps to Reproduce
1. Start Docker: `docker compose up`
2. Upload whale_001.jpg via frontend
3. Error appears

## Error Message
```
Internal Server Error: Model not found
```

## Environment
- OS: Ubuntu 22.04
- Python: 3.11.6
- Docker: 24.0.5
- Model: Downloaded from Hugging Face

## Expected
Prediction result with species name

## Actual
500 error
```

---

**Last Updated:** September 1, 2025
**Version:** 0.1.0
