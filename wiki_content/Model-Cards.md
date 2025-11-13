# Model Cards

Detailed specifications, performance metrics, and usage guidelines for all models in the Whales Identification project.

---

## Table of Contents

- [Model Comparison](#model-comparison)
- [Vision Transformer L/32](#vision-transformer-l32)
- [Vision Transformer B/16](#vision-transformer-b16)
- [EfficientNet-B5](#efficientnet-b5)
- [EfficientNet-B0](#efficientnet-b0)
- [ResNet-101](#resnet-101)
- [ResNet-54](#resnet-54)
- [Swin Transformer](#swin-transformer)
- [Training Details](#training-details)
- [Evaluation Methodology](#evaluation-methodology)

---

## Model Comparison

### Performance Summary

| Model | Precision@1 | Inference Time | Parameters | Model Size | Status |
|-------|-------------|---------------|------------|------------|--------|
| **Vision Transformer L/32** | **93%** | ~3.5s | 307M | 1.2 GB | ⭐ Best Accuracy |
| Vision Transformer B/16 | 91% | ~2.0s | 86M | 340 MB | ✅ Production |
| EfficientNet-B5 | 91% | ~1.8s | 30M | 120 MB | ✅ Production |
| **EfficientNet-B0** | 88% | **~1.0s** | 5.3M | 21 MB | ⚡ Fastest |
| ResNet-101 | 85% | ~1.2s | 44M | 170 MB | ✅ Baseline |
| ResNet-54 | 82% | ~0.8s | 25M | 100 MB | ⚡ Fastest CNN |
| Swin Transformer | 90% | ~2.2s | 88M | 350 MB | 🔬 Research |

**Hardware:** Measurements on single NVIDIA Tesla V100 GPU, batch size 1

### Trade-offs Matrix

```
Accuracy vs Speed:
  High ──┐
         │                    ViT-L/32 ●
         │
         │         ViT-B/16 ●    Swin ●
Precision│      EfficientNet-B5 ●
         │
         │           ResNet-101 ●
         │                 EfficientNet-B0 ●
         │                       ResNet-54 ●
   Low ──┴──────────────────────────────────────▶
         Slow                                  Fast
                   Inference Time
```

---

## Vision Transformer L/32

### Model Overview

**Architecture:** Vision Transformer Large with 32×32 patch size
**Backbone:** `timm.vit_large_patch32_224`
**Status:** Best accuracy, recommended for research and high-precision applications

### Specifications

| Attribute | Value |
|-----------|-------|
| **Input Size** | 448×448×3 |
| **Patch Size** | 32×32 |
| **Embedding Dim** | 1024 |
| **Depth** | 24 layers |
| **Attention Heads** | 16 |
| **Parameters** | 307M |
| **Model File** | model-e15.pt (2.1 GB with optimizer state) |
| **Training Dataset** | HappyWhale + Ministry RF (~60,000 images) |
| **Classes** | 15,587 individual whales |

### Performance Metrics

#### Overall Performance

| Metric | Value |
|--------|-------|
| **Precision@1** | 93.2% |
| **Precision@5** | 97.8% |
| **Recall** | 91.5% |
| **F1-Score** | 0.923 |
| **mAP** | 0.915 |
| **Inference Time** | 3.5s (V100), 12s (CPU) |

#### Per-Species Performance (Top 10)

| Species | Precision | Recall | F1 | Sample Count |
|---------|-----------|--------|----|----|
| Humpback Whale | 95.3% | 93.8% | 0.945 | 12,543 |
| Blue Whale | 94.1% | 92.5% | 0.933 | 8,721 |
| Fin Whale | 92.8% | 91.2% | 0.920 | 6,432 |
| Gray Whale | 93.5% | 90.8% | 0.921 | 5,124 |
| Beluga Whale | 91.2% | 89.5% | 0.903 | 3,856 |
| Right Whale | 90.7% | 88.3% | 0.895 | 2,945 |
| Sperm Whale | 89.5% | 87.1% | 0.883 | 2,134 |
| Orca | 94.8% | 93.2% | 0.940 | 1,832 |
| Bottlenose Dolphin | 88.3% | 86.7% | 0.875 | 1,523 |
| Spinner Dolphin | 87.1% | 84.9% | 0.860 | 1,234 |

### Intended Use

**Recommended for:**
- ✅ Research applications requiring highest accuracy
- ✅ Offline batch processing
- ✅ High-value species identification
- ✅ Dataset validation and annotation

**Not recommended for:**
- ❌ Real-time applications (<1s latency)
- ❌ Edge devices (large model size)
- ❌ Mobile deployment

### Limitations

- **Speed:** 3.5s inference time may be too slow for real-time
- **Memory:** Requires 4GB+ GPU memory for batch processing
- **Robustness:** 15-20% accuracy drop on:
  - Low-resolution images (<800×600)
  - Heavy occlusion (>50% whale hidden)
  - Extreme weather conditions (fog, rain)
  - Night-time images with poor lighting

### Training Details

```yaml
Hyperparameters:
  epochs: 15
  batch_size: 32
  learning_rate: 1e-4
  optimizer: AdamW
  weight_decay: 1e-4
  scheduler: CosineAnnealingLR
  loss: CrossEntropyLoss + ArcFace (m=0.5, s=30)
  augmentation: Albumentations (flip, rotate, color jitter)

Training Time: ~48 hours on 4x V100 GPUs
Final Loss: 0.234 (train), 0.412 (val)
Best Epoch: 15
Checkpoint: models/model-e15.pt
```

---

## Vision Transformer B/16

### Model Overview

**Architecture:** Vision Transformer Base with 16×16 patch size
**Backbone:** `timm.vit_base_patch16_224`
**Status:** Production-ready, currently deployed in API

### Specifications

| Attribute | Value |
|-----------|-------|
| **Input Size** | 448×448×3 |
| **Patch Size** | 16×16 |
| **Embedding Dim** | 768 |
| **Depth** | 12 layers |
| **Attention Heads** | 12 |
| **Parameters** | 86M |
| **Model Size** | 340 MB |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Precision@1** | 91.3% |
| **Precision@5** | 96.1% |
| **Recall** | 89.8% |
| **F1-Score** | 0.905 |
| **Inference Time** | 2.0s (V100), 7s (CPU) |

### Intended Use

**Recommended for:**
- ✅ Production API deployments
- ✅ Batch processing (10-100 images)
- ✅ High-throughput applications
- ✅ GPU servers

**Balanced trade-off:** Good accuracy with reasonable speed

---

## EfficientNet-B5

### Model Overview

**Architecture:** EfficientNet-B5 with compound scaling
**Backbone:** `timm.efficientnet_b5`
**Status:** Production-ready, alternative to ViT-B/16

### Specifications

| Attribute | Value |
|-----------|-------|
| **Input Size** | 456×456×3 |
| **Depth** | Deep (multiple blocks) |
| **Width Multiplier** | 1.6 |
| **Parameters** | 30M |
| **Model Size** | 120 MB |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Precision@1** | 91.0% |
| **Precision@5** | 95.8% |
| **Recall** | 89.2% |
| **F1-Score** | 0.901 |
| **Inference Time** | 1.8s (V100), 6s (CPU) |

### Intended Use

**Recommended for:**
- ✅ Environments with limited GPU memory
- ✅ Mobile GPU deployment (Snapdragon, Mali)
- ✅ Faster inference than ViT with similar accuracy

**Advantages over ViT:**
- Smaller model size (120 MB vs 340 MB)
- More efficient on CPU

---

## EfficientNet-B0

### Model Overview

**Architecture:** EfficientNet-B0 (smallest variant)
**Backbone:** `timm.efficientnet_b0`
**Status:** Production-ready for real-time applications

### Specifications

| Attribute | Value |
|-----------|-------|
| **Input Size** | 224×224×3 |
| **Parameters** | 5.3M |
| **Model Size** | 21 MB |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Precision@1** | 88.1% |
| **Precision@5** | 94.3% |
| **Recall** | 86.5% |
| **F1-Score** | 0.873 |
| **Inference Time** | 1.0s (V100), 3s (CPU) |

### Intended Use

**Recommended for:**
- ✅ **Real-time applications** (target: <2s latency)
- ✅ **Edge devices** (Jetson Nano, Coral)
- ✅ **Mobile apps** (iOS, Android)
- ✅ **High-throughput batch processing** (>100 images)

**Trade-off:** 5% accuracy drop for 3.5× speedup vs ViT-L/32

### Deployment Example

```python
# Mobile-optimized inference
import torch
import torch.quantization

# Load model
model = EfficientNetB0.load_pretrained()

# Quantize for mobile
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Export to ONNX
torch.onnx.export(model_quantized, dummy_input, "efficientnet_b0.onnx")

# Inference time: ~300ms on Snapdragon 888
```

---

## ResNet-101

### Model Overview

**Architecture:** ResNet-101 (Deep Residual Network)
**Backbone:** `torchvision.models.resnet101`
**Status:** Baseline comparison model

### Specifications

| Attribute | Value |
|-----------|-------|
| **Input Size** | 224×224×3 |
| **Depth** | 101 layers |
| **Parameters** | 44M |
| **Model Size** | 170 MB |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Precision@1** | 85.3% |
| **Precision@5** | 92.7% |
| **Recall** | 83.8% |
| **F1-Score** | 0.845 |
| **Inference Time** | 1.2s (V100), 4s (CPU) |

### Intended Use

**Recommended for:**
- ✅ Baseline comparisons
- ✅ Legacy system integrations
- ✅ Transfer learning experiments

**Note:** Lower accuracy than ViT and EfficientNet, but well-established architecture

---

## ResNet-54

### Model Overview

**Architecture:** ResNet-54 (lighter variant)
**Backbone:** Custom ResNet implementation
**Status:** Fastest CNN for edge deployment

### Specifications

| Attribute | Value |
|-----------|-------|
| **Input Size** | 224×224×3 |
| **Depth** | 54 layers |
| **Parameters** | 25M |
| **Model Size** | 100 MB |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Precision@1** | 82.4% |
| **Precision@5** | 90.8% |
| **Recall** | 80.9% |
| **F1-Score** | 0.816 |
| **Inference Time** | 0.8s (V100), 2.5s (CPU) |

### Intended Use

**Recommended for:**
- ✅ Ultra-fast screening (pre-filtering)
- ✅ Resource-constrained environments
- ✅ Edge devices with limited compute

**Trade-off:** Lowest accuracy, but fastest inference

---

## Swin Transformer

### Model Overview

**Architecture:** Swin Transformer (Shifted Windows)
**Backbone:** `timm.swin_base_patch4_window7_224`
**Status:** Research model, experimental

### Specifications

| Attribute | Value |
|-----------|-------|
| **Input Size** | 224×224×3 |
| **Window Size** | 7×7 |
| **Patch Size** | 4×4 |
| **Parameters** | 88M |
| **Model Size** | 350 MB |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Precision@1** | 90.2% |
| **Precision@5** | 95.5% |
| **Recall** | 88.7% |
| **F1-Score** | 0.894 |
| **Inference Time** | 2.2s (V100), 8s (CPU) |

### Intended Use

**Recommended for:**
- 🔬 Research experiments
- 🔬 Hierarchical feature extraction
- 🔬 Multi-scale analysis

**Not production-ready:** Requires further validation

---

## Training Details

### Common Training Configuration

**Dataset:**
- Source: HappyWhale (CC-BY-NC-4.0) + Ministry RF (research-only)
- Total images: ~60,000 train, ~20,000 test
- Classes: 15,587 individual whales
- Split: 70% train, 15% validation, 15% test

**Augmentation Pipeline (Albumentations):**

```python
train_transform = A.Compose([
    A.RandomResizedCrop(height=448, width=448, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**Optimizer Configuration:**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=15,
    eta_min=1e-6
)
```

**Loss Function:**

```python
# ArcFace loss with CrossEntropy
loss = ArcFaceLoss(
    in_features=512,
    out_features=15587,
    scale=30.0,
    margin=0.50
)
```

---

## Evaluation Methodology

### Metrics Definitions

**Precision@1:**
```
Precision@1 = (Correct top-1 predictions) / (Total predictions)
```

**Precision@5:**
```
Precision@5 = (Predictions where true label in top-5) / (Total predictions)
```

**Recall:**
```
Recall = (True Positives) / (True Positives + False Negatives)
```

**F1-Score:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Test Set

- **Size:** 20,000 images
- **Distribution:** Balanced across top 100 species, long-tail for rare species
- **Quality:** High-resolution (≥1920×1080), clear weather conditions

### Inference Benchmarking

**Hardware:**
- GPU: NVIDIA Tesla V100 (16GB)
- CPU: Intel Xeon Gold 6154 (18 cores)
- RAM: 64GB

**Protocol:**
1. Warm-up: 10 inference runs
2. Measurement: 100 runs, report mean ± std
3. Batch size: 1 (single image latency)

---

## Model Selection Guide

### Decision Tree

```
Start: What's your priority?
│
├─ Highest Accuracy?
│  └─▶ Vision Transformer L/32 (93%)
│
├─ Production API?
│  ├─ GPU available?
│  │  └─▶ Vision Transformer B/16 (91%, 2.0s)
│  └─ CPU only?
│     └─▶ EfficientNet-B5 (91%, 6s CPU)
│
├─ Real-time (<2s)?
│  └─▶ EfficientNet-B0 (88%, 1.0s)
│
└─ Edge Device?
   ├─ Mobile GPU?
   │  └─▶ EfficientNet-B0 quantized (88%, ~300ms)
   └─ Jetson Nano?
      └─▶ ResNet-54 (82%, 0.8s)
```

---

## Future Improvements

**Planned Enhancements:**
- ✅ ConvNeXt models (similar to Swin but faster)
- ✅ Model distillation (ViT-L/32 → EfficientNet-B0)
- ✅ Ensemble methods (ViT + EfficientNet)
- ✅ ONNX Runtime optimization
- ✅ TensorRT deployment

---

**Related Pages:**
- [Architecture](Architecture) - Technical implementation details
- [Testing](Testing) - Model evaluation procedures
- [Usage](Usage) - How to use each model
