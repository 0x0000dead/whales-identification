# Architecture

System architecture and technical design of the Whales Identification project.

---

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [ML Pipeline](#ml-pipeline)
- [API Design](#api-design)
- [Deployment Architecture](#deployment-architecture)
- [Technology Stack](#technology-stack)

---

## System Overview

**Project Name:** Whales Identification (EcoMarineAI)

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                    │
├────────────────┬────────────────────────────────────────────────┤
│  React Frontend│  Streamlit Demo  │  Jupyter Notebooks │  API  │
│  (Port 8080)   │  (Port 8501)     │  (Local)           │ Docs  │
└────────────────┴──────────────────┴────────────────────────────┬┘
                                                                  │
                         HTTP/REST API                            │
                                                                  │
┌─────────────────────────────────────────────────────────────────▼┐
│                      FastAPI Backend (Port 8000)                 │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Routers    │  │   Response   │  │   Whale      │          │
│  │  (Endpoints) │─▶│    Models    │─▶│  Inference   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────┬────────────────────┘
                                              │
                           Model Loading      │
                                              │
┌─────────────────────────────────────────────▼────────────────────┐
│                      ML Inference Layer                           │
├───────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  Vision Trans-   │  │  Background      │  │  Config        │ │
│  │  former Model    │  │  Removal (rembg) │  │  (1,000 IDs)   │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                                  │
                                  │ PyTorch
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│                      Storage Layer                                 │
├────────────────────────────────────────────────────────────────────┤
│  models/model-e15.pt  │  whales_be_service/config.yaml           │
│  (2.1 GB checkpoint)  │  (ID → Species mapping, 1,000 whales)    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Frontend (React + TypeScript)

**Directory:** `frontend/`

#### Components

```
frontend/
├── src/
│   ├── App.tsx                  # Main application component
│   ├── components/
│   │   ├── ImageUpload.tsx      # Single image upload
│   │   ├── BatchUpload.tsx      # ZIP batch upload
│   │   ├── ResultDisplay.tsx    # Prediction results
│   │   ├── ErrorModal.tsx       # Error handling
│   │   └── LoadingSpinner.tsx   # Loading state
│   ├── api/
│   │   └── client.ts            # API client (fetch wrappers)
│   ├── types/
│   │   └── detection.ts         # TypeScript interfaces
│   └── utils/
│       └── imageProcessing.ts   # Image utilities
├── public/
│   └── assets/
└── vite.config.ts               # Vite configuration
```

#### Key Technologies

- **React 18:** UI framework with hooks
- **TypeScript:** Type-safe development
- **Vite:** Fast build tool and dev server
- **Recharts:** Data visualization
- **Axios/Fetch:** HTTP client

#### Design Patterns

- **Component Composition:** Reusable UI components
- **State Management:** React useState/useEffect
- **Error Boundaries:** Graceful error handling
- **Lazy Loading:** Code splitting for performance

---

### 2. Backend (FastAPI + Python)

**Directory:** `whales_be_service/`

#### Structure

```
whales_be_service/
├── src/whales_be_service/
│   ├── main.py                  # FastAPI app + CORS setup
│   ├── routers.py               # API route definitions
│   ├── response_models.py       # Pydantic models + inference logic
│   ├── whale_infer.py           # Model loading and inference
│   └── config.yaml              # 1,000 whale ID → species
├── tests/
│   └── api/
│       └── test_post_endpoints.py  # Integration tests
└── pyproject.toml               # Poetry dependencies
```

#### Key Classes

**`WhaleInference` (whale_infer.py)**

```python
class WhaleInference:
    def __init__(self, model_path: str, config_path: str):
        self.model = self.load_model(model_path)
        self.id_to_species = self.load_config(config_path)

    def predict(self, image: np.ndarray) -> Detection:
        # 1. Preprocess image (resize, normalize)
        tensor = self.preprocess(image)

        # 2. Model inference
        with torch.no_grad():
            embeddings = self.model(tensor)
            logits = self.model.classify(embeddings)

        # 3. Postprocess (top-1, species lookup)
        individual_id = self.get_top_prediction(logits)
        species = self.id_to_species[individual_id]

        # 4. Background removal
        mask = rembg.remove(image)

        return Detection(
            class_animal=individual_id,
            id_animal=species,
            probability=confidence,
            mask=base64_encode(mask)
        )
```

**`Detection` (response_models.py)**

```python
class Detection(BaseModel):
    image_ind: str              # Filename
    bbox: list[int]            # [x, y, width, height]
    class_animal: str          # Individual ID
    id_animal: str             # Species name
    probability: float         # Confidence (0.0-1.0)
    mask: str | None          # Base64 PNG
```

#### API Endpoints

**`routers.py`**

```python
@router.post("/predict-single", response_model=Detection)
async def predict_single(file: UploadFile = File(...)):
    # 1. Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Unsupported media type")

    # 2. Read image
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 3. Inference
    result = whale_inference.predict(image)

    return result

@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    # 1. Validate ZIP
    if not zipfile.is_zipfile(file.file):
        raise HTTPException(400, "Invalid ZIP file")

    # 2. Extract images
    with zipfile.ZipFile(file.file, 'r') as zip_ref:
        images = [img for img in zip_ref.namelist() if img.endswith(('.jpg', '.png'))]

    # 3. Batch inference
    results = []
    for image_path in images:
        image = zip_ref.read(image_path)
        result = whale_inference.predict(image)
        results.append(result)

    return {"results": results, "total_processed": len(results)}
```

---

### 3. ML Core (whales_identify)

**Directory:** `whales_identify/`

#### Components

```
whales_identify/
├── model.py                     # HappyWhaleModel architecture
├── dataset.py                   # WhaleDataset (PyTorch Dataset)
├── train.py                     # Training loop
├── config.py                    # Training hyperparameters
└── utils/
    ├── augmentation.py          # Albumentations transforms
    └── metrics.py               # Evaluation metrics
```

#### Model Architecture

**`HappyWhaleModel` (model.py)**

```python
class HappyWhaleModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int = 1000, embedding_size: int = 512):
        super().__init__()

        # 1. Backbone (Vision Transformer or CNN)
        self.backbone = timm.create_model(backbone, pretrained=True)

        # 2. GeM Pooling (better than average pooling for metric learning)
        self.gem_pooling = GeM()

        # 3. Embedding layer
        self.fc_embedding = nn.Linear(self.backbone.num_features, embedding_size)

        # 4. ArcFace head (metric learning) - 1,000 whales and dolphins
        self.arcface = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,  # 1,000 individuals
            s=30.0,  # Scale
            m=0.50   # Margin
        )

    def forward(self, x):
        # 1. Extract features
        features = self.backbone(x)

        # 2. GeM pooling
        pooled = self.gem_pooling(features)

        # 3. Embedding
        embeddings = self.fc_embedding(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def classify(self, embeddings, labels=None):
        # ArcFace classification
        logits = self.arcface(embeddings, labels)
        return logits
```

**GeM Pooling:**
```python
class GeM(nn.Module):
    """Generalized Mean Pooling - better than average for metric learning"""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
```

**ArcFace Margin:**
```python
class ArcMarginProduct(nn.Module):
    """Additive Angular Margin for metric learning"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # Cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Add margin
        if label is not None:
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            output = cosine * self.s

        return output
```

---

## Data Flow

### Single Image Prediction

```
1. User uploads image (JPG/PNG)
   └─▶ Frontend: ImageUpload.tsx

2. FormData POST to /predict-single
   └─▶ Frontend: api/client.ts

3. Backend receives file
   └─▶ Backend: routers.py

4. Validate content type
   └─▶ Backend: routers.py (check MIME)

5. Read image bytes
   └─▶ OpenCV: cv2.imdecode()

6. Preprocess image
   ├─ Resize to 448×448
   ├─ Normalize (ImageNet stats)
   └─ Convert to tensor
   └─▶ whale_infer.py

7. Model inference
   ├─ Backbone → embeddings (512-dim)
   ├─ ArcFace → logits (1,000-dim)
   └─ Softmax → probabilities
   └─▶ PyTorch model

8. Postprocess
   ├─ Top-1 prediction → individual ID
   ├─ Lookup species in config.yaml
   └─ Background removal (rembg)
   └─▶ response_models.py

9. Return JSON
   └─▶ Detection object

10. Frontend displays result
    ├─ Species name
    ├─ Confidence bar
    ├─ Bounding box overlay
    └─ Mask image
    └─▶ ResultDisplay.tsx
```

### Batch Processing

```
1. User creates ZIP with images
   └─▶ Local file system

2. Upload ZIP to /predict-batch
   └─▶ Frontend: BatchUpload.tsx

3. Backend extracts ZIP
   └─▶ Python: zipfile.ZipFile

4. Iterate over images
   └─▶ For each image: steps 5-8 from single prediction

5. Collect results
   └─▶ List[Detection]

6. Return batch response
   └─▶ {"results": [...], "total_processed": N}

7. Frontend displays table
   └─▶ ResultTable.tsx with pagination
```

---

## ML Pipeline

### Training Pipeline

```
1. Data Preparation
   ├─ Download HappyWhale dataset (Kaggle)
   ├─ Download Ministry RF dataset
   ├─ Merge and deduplicate
   └─ Train/val/test split (70/15/15)

2. Data Augmentation (Albumentations)
   ├─ RandomResizedCrop(448, 448)
   ├─ HorizontalFlip(p=0.5)
   ├─ ShiftScaleRotate(p=0.5)
   ├─ HueSaturationValue(p=0.3)
   ├─ RandomBrightnessContrast(p=0.3)
   └─ Normalize(ImageNet stats)

3. Model Initialization
   ├─ Backbone: timm.create_model("vit_large_patch32_224")
   ├─ Pretrained: ImageNet weights
   ├─ GeM pooling
   └─ ArcFace head (1,000 classes)

4. Training Loop (15 epochs)
   ├─ Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
   ├─ Scheduler: CosineAnnealingLR
   ├─ Loss: CrossEntropyLoss (with ArcFace margin)
   ├─ Batch size: 32
   └─ Checkpointing: Save best model by val accuracy

5. Evaluation
   ├─ Metrics: Precision@1, Precision@5, Recall, F1
   ├─ Confusion matrix
   └─ Per-species performance

6. Model Export
   └─ Save checkpoint: {epoch, model_state_dict, optimizer_state_dict, loss}
```

### Inference Pipeline

```
1. Load Model
   ├─ torch.load("models/model-e15.pt")
   ├─ model.load_state_dict(checkpoint['model_state_dict'])
   └─ model.eval()

2. Preprocess
   ├─ cv2.imread() or PIL.Image.open()
   ├─ Resize to 448×448
   ├─ Normalize: (x - mean) / std
   └─ ToTensor(): [H, W, C] → [C, H, W]

3. Inference
   ├─ with torch.no_grad():
   │   ├─ embeddings = model(tensor)
   │   └─ logits = model.classify(embeddings)
   ├─ probabilities = F.softmax(logits, dim=1)
   └─ top_k = torch.topk(probabilities, k=5)

4. Postprocess
   ├─ individual_id = int(top_k.indices[0])
   ├─ species = config.yaml[individual_id]
   ├─ confidence = float(top_k.values[0])
   └─ mask = rembg.remove(original_image)

5. Return
   └─ Detection(class_animal, id_animal, probability, mask)
```

---

## API Design

### RESTful Principles

- **Stateless:** Each request contains all necessary info
- **Resource-based:** Endpoints represent resources (/predict-single, /predict-batch)
- **HTTP methods:** POST for predictions (side effects: model inference)
- **Status codes:** 200 (success), 400 (bad request), 500 (server error)

### Request/Response Flow

```
Client Request
   │
   ├─ Headers: Content-Type: multipart/form-data
   ├─ Body: file=<binary image data>
   └─ Method: POST
   │
   ▼
FastAPI Router
   │
   ├─ Validate content type
   ├─ Check file size
   └─ Parse multipart data
   │
   ▼
Inference Engine
   │
   ├─ Load image
   ├─ Preprocess
   ├─ Model inference
   └─ Postprocess
   │
   ▼
Response Model (Pydantic)
   │
   ├─ Validate fields
   ├─ Serialize to JSON
   └─ Add headers
   │
   ▼
HTTP Response
   │
   ├─ Status: 200 OK
   ├─ Content-Type: application/json
   └─ Body: {"image_ind": "...", "id_animal": "...", ...}
```

---

## Deployment Architecture

### Docker Compose Stack

```yaml
version: "3.8"

services:
  backend:
    build: ./whales_be_service
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model-e15.pt
      - CONFIG_PATH=/app/whales_be_service/config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/docs"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    ports:
      - "8080:80"
    depends_on:
      - backend
    environment:
      - VITE_BACKEND_URL=http://backend:8000

networks:
  whale-net:
    driver: bridge
```

### Scaling Strategy

**Horizontal Scaling:**
```
┌──────────────┐
│ Load Balancer│ (Nginx/HAProxy)
└──────┬───────┘
       │
    ┌──┴──┬──────┬──────┐
    │     │      │      │
┌───▼─┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐
│API 1│ │API 2│ │API 3│ │API N│
└─────┘ └────┘ └────┘ └────┘
```

**Considerations:**
- Stateless API (no session storage)
- Shared model storage (NFS or S3)
- Redis for caching predictions

---

## Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | FastAPI | 0.115+ | REST API framework |
| **Server** | Uvicorn | 0.34+ | ASGI server |
| **ML** | PyTorch | 2.4.1 | Deep learning |
| **Models** | TIMM | 1.0.9 | Pretrained models |
| **Vision** | OpenCV | 4.10+ | Image processing |
| **Validation** | Pydantic | 2.9+ | Data validation |
| **Background Removal** | rembg | 2.0+ | U²-Net segmentation |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | React | 18.x | UI library |
| **Language** | TypeScript | 5.x | Type-safe JS |
| **Build Tool** | Vite | 5.x | Fast dev server |
| **Charts** | Recharts | 2.x | Data visualization |
| **HTTP Client** | Axios | 1.x | API requests |

### DevOps

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Containerization** | Docker | 20.10+ | Containerization |
| **Orchestration** | Docker Compose | 2.0+ | Multi-container |
| **CI/CD** | GitHub Actions | N/A | Automation |
| **Package Manager** | Poetry | 1.5+ | Python deps |
| **Pre-commit** | pre-commit | 3.x | Git hooks |

---

**Next Steps:**
- [Model Cards](Model-Cards) - Detailed model specifications
- [Testing](Testing) - Testing architecture
- [API Reference](API-Reference) - Complete API docs
