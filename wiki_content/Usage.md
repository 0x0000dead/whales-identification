# Usage Guide

Learn how to use Whales Identification in different scenarios.

---

## Table of Contents

- [Frontend UI](#frontend-ui)
- [REST API](#rest-api)
- [Streamlit Demo](#streamlit-demo)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Python Library](#python-library)
- [Best Practices](#best-practices)

---

## Frontend UI

### Quick Start

1. **Start services:**
```bash
docker compose up --build
```

2. **Open browser:**
- Frontend: http://localhost:8080
- API Docs: http://localhost:8000/docs

### Single Image Upload

#### Step 1: Select Image

Click "Choose File" or drag-and-drop image onto upload zone.

**Supported formats:**
- JPG, JPEG, PNG
- Recommended: 1920×1080 or higher
- Max size: 10 MB

#### Step 2: Preview

Image preview displays automatically.

#### Step 3: Submit

Click "Identify" button.

**Processing time:** 1-4 seconds depending on model and image size.

#### Step 4: View Results

Results display:
- **Species name** (e.g., "Humpback Whale")
- **Individual ID** (hex string)
- **Confidence score** (0-100%)
- **Bounding box** overlay
- **Background-removed mask**

### Batch Processing

#### Step 1: Prepare ZIP

Create ZIP archive with images:

```bash
# Linux/macOS
zip whale_batch.zip whale_*.jpg

# Windows
# Use File Explorer: Select files → Right-click → Send to → Compressed folder
```

**Requirements:**
- Max 100 images per batch
- Max ZIP size: 50 MB
- Only JPG, PNG, JPEG inside

#### Step 2: Upload ZIP

1. Switch to "Batch Mode" tab
2. Select ZIP file
3. Click "Process Batch"

#### Step 3: View Results

Results table shows:
- Image filename
- Species
- Individual ID
- Confidence
- Thumbnails

**Export options:**
- Download results as CSV
- Download individual masks
- Download summary report

---

## REST API

### Single Image Prediction

#### Using curl

```bash
curl -X POST "http://localhost:8000/predict-single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@whale_image.jpg"
```

#### Using Python requests

```python
import requests

url = "http://localhost:8000/predict-single"

with open("whale_image.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()
print(f"{result['id_animal']}: {result['probability']:.2%}")
```

**Full example:** See [API Reference](API-Reference#post-predict-single)

### Batch Processing

```bash
# Create ZIP
zip batch.zip whale_*.jpg

# Upload
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/zip" \
  --data-binary "@batch.zip"
```

**Full example:** See [API Reference](API-Reference#post-predict-batch)

---

## Streamlit Demo

### Best Model Demo (Vision Transformer)

```bash
cd research/demo-ui
poetry install
poetry run streamlit run streamlit_app.py --server.port=8501
```

**Features:**
- Single image upload
- Real-time inference
- Visualization of predictions
- Probability distribution chart

### Alternative Demo (With Masking)

```bash
cd research/demo-ui-mask
poetry install
poetry run streamlit run streamlit_app.py --server.port=8502
```

**Features:**
- Manual mask drawing
- Background removal preview
- Improved accuracy with masks

---

## Jupyter Notebooks

### Research Notebooks

Located in `research/notebooks/`:

#### 1. Vision Transformer Training

**Notebook:** `02_ViT_train_effiecientnet.ipynb`

```python
# Launch Jupyter
cd research/notebooks
jupyter notebook

# Open 02_ViT_train_effiecientnet.ipynb
```

**Contents:**
- Data preparation
- Model architecture setup
- Training loop with metric learning
- Checkpoint saving

#### 2. Vision Transformer Inference

**Notebook:** `02_ViT_inference_efficientnet.ipynb`

**Contents:**
- Model loading
- Inference on test set
- Metrics calculation
- Visualization of results

#### 3. EfficientNet Experiments

**Notebook:** `03_efficientnet_experiments.ipynb`

**Contents:**
- Comparison of EfficientNet-B0, B3, B5
- Metric learning approach
- Speed vs accuracy trade-offs

#### 4. ResNet Classification

**Notebook:** `04_resnet_classification_experiments.ipynb`

**Contents:**
- ResNet-54 and ResNet-101
- Traditional classification approach
- Comparison with metric learning

#### 5. Swin Transformer

**Notebook:** `05_swinT_experiments.ipynb`

**Contents:**
- Swin Transformer architecture
- Performance evaluation
- Comparison with ViT

#### 6. Binary Benchmark

**Notebook:** `06_benchmark_binary.ipynb`

**Contents:**
- Binary classification (whale present/absent)
- All models comparison
- ROC curves, precision-recall

#### 7. Multiclass Benchmark

**Notebook:** `06_benchmark_multiclass.ipynb`

**Contents:**
- Multiclass species identification
- Confusion matrices
- Per-species performance

#### 8. ONNX Inference

**Notebook:** `07_onnx_inference_compare.ipynb`

**Contents:**
- ONNX conversion
- Speed comparison
- Deployment optimization

---

## Python Library

### Using whales_identify Module

#### Training a Model

```python
from whales_identify.train import train_model
from whales_identify.dataset import WhaleDataset
from whales_identify.model import HappyWhaleModel

# Prepare dataset
dataset = WhaleDataset(
    csv_path="data/train.csv",
    image_dir="data/images",
    transform=get_train_transform()
)

# Initialize model
model = HappyWhaleModel(
    backbone="efficientnet_b0",
    num_classes=1000,  # 1,000 individual whales and dolphins
    embedding_size=512
)

# Train
train_model(
    model=model,
    dataset=dataset,
    epochs=15,
    batch_size=32,
    lr=0.001
)
```

#### Inference

```python
from whales_identify.model import HappyWhaleModel
import torch
from PIL import Image

# Load model
model = HappyWhaleModel.load_from_checkpoint("models/model-e15.pt")
model.eval()

# Load image
image = Image.open("whale.jpg")
tensor = preprocess(image)

# Predict
with torch.no_grad():
    embeddings = model(tensor)
    prediction = model.classify(embeddings)

print(f"Individual ID: {prediction['class_animal']}")
print(f"Confidence: {prediction['probability']}")
```

### Using Backend API Module

```python
from whales_be_service.whale_infer import WhaleInference

# Initialize inference engine
inferencer = WhaleInference(
    model_path="models/model-e15.pt",
    config_path="whales_be_service/config.yaml"
)

# Predict single image
result = inferencer.predict_image("whale.jpg")

# Batch predict
results = inferencer.predict_batch([
    "whale_001.jpg",
    "whale_002.jpg",
    "whale_003.jpg"
])
```

---

## Best Practices

### Image Quality

**Recommended:**
- Resolution: ≥1920×1080
- Format: JPG (smaller size) or PNG (lossless)
- Lighting: Good natural lighting
- Angle: Aerial view, ~30-45° angle
- Distance: Whale fills 20-80% of frame

**Avoid:**
- Blurry images
- Heavy shadows
- Extreme angles
- Cropped whales
- Low resolution (<800×600)

### Batch Processing

**Optimal batch sizes:**
- Small batches (1-10 images): <1 minute
- Medium batches (10-50 images): 1-5 minutes
- Large batches (50-100 images): 5-15 minutes

**Tips:**
- Use ZIP compression for faster uploads
- Process overnight for large datasets
- Monitor memory usage (8GB RAM minimum)

### Model Selection

| Use Case | Recommended Model | Speed | Accuracy |
|----------|------------------|-------|----------|
| **Production API** | Vision Transformer B/16 | 2.0s | 91% |
| **Research** | Vision Transformer L/32 | 3.5s | 93% |
| **Real-time** | EfficientNet-B0 | 1.0s | 88% |
| **Balanced** | EfficientNet-B5 | 1.8s | 91% |

**See [Model Cards](Model-Cards) for detailed comparison.**

### Error Handling

```python
import requests

try:
    response = requests.post(url, files=files, timeout=30)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("Request timed out. Try smaller image.")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        print("Invalid image format")
    elif e.response.status_code == 500:
        print("Server error. Contact support.")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Optimization

#### Server-side

```python
# Enable GPU inference
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Batch processing
images = load_images(["img1.jpg", "img2.jpg", "img3.jpg"])
tensor_batch = torch.stack([preprocess(img) for img in images])
predictions = model(tensor_batch)
```

#### Client-side

```python
# Resize images before upload
from PIL import Image

def optimize_image(image_path, max_size=1920):
    img = Image.open(image_path)

    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)

    # Save as JPG with quality=85
    img.save("optimized.jpg", "JPEG", quality=85)
```

---

## Common Workflows

### Workflow 1: Single Image Quick Check

```bash
# 1. Upload image via frontend
http://localhost:8080

# 2. View result
# Species, confidence, mask displayed

# 3. Download mask
# Right-click mask → Save image
```

### Workflow 2: Batch Research Analysis

```bash
# 1. Organize images
mkdir whale_images
cp /path/to/dataset/*.jpg whale_images/

# 2. Create ZIP
zip -r whale_batch.zip whale_images/

# 3. Process batch
curl -X POST "http://localhost:8000/predict-batch" \
  --data-binary "@whale_batch.zip" \
  > results.json

# 4. Analyze results
python analyze_results.py results.json
```

### Workflow 3: Model Experimentation

```bash
# 1. Launch Jupyter
cd research/notebooks
jupyter notebook

# 2. Open notebook
# 02_ViT_train_effiecientnet.ipynb

# 3. Modify hyperparameters
# batch_size, learning_rate, architecture

# 4. Train model
# Execute all cells

# 5. Evaluate
# Open 02_ViT_inference_efficientnet.ipynb
```

---

## Monitoring and Logging

### Backend Logs

```bash
# View real-time logs
docker compose logs -f backend

# Filter for errors
docker compose logs backend | grep ERROR

# Save logs to file
docker compose logs backend > backend.log
```

### Frontend Logs

```bash
# View frontend logs
docker compose logs -f frontend

# Browser console
# Open DevTools (F12) → Console tab
```

---

## Next Steps

- **[API Reference](API-Reference)** - Complete API documentation
- **[Architecture](Architecture)** - Understand system design
- **[Model Cards](Model-Cards)** - Model performance details
- **[Testing](Testing)** - Run tests and verify functionality

---

**Need help?** [Open an issue](https://github.com/0x0000dead/whales-identification/issues) or check [FAQ](FAQ).
