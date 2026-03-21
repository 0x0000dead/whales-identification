# Model Card: EcoMarineAI Whale Identification

## Model Details

- **Model Name:** EcoMarineAI Vision Transformer (ViT-L/32)
- **Version:** 1.0.0
- **Type:** Multi-class classification with metric learning (ArcFace)
- **Architecture:** Vision Transformer Large, patch size 32
- **Framework:** PyTorch 2.4.1
- **License:** Apache 2.0 with restrictions (see LICENSE_MODELS.md)
- **Repository:** [baltsat/Whales-Identification](https://huggingface.co/baltsat/Whales-Identification)

## Intended Use

- **Primary Use:** Automated identification of individual marine mammals (whales and dolphins) from aerial photography
- **Users:** Marine biologists, ecology researchers, government environmental agencies, conservation organizations
- **Out-of-scope:** Real-time video processing, underwater photography, species not in the training set

## Training Data

- **Source:** HappyWhale community dataset + Ministry of Natural Resources and Ecology of RF
- **Size:** ~60,000 training images, ~20,000 test images
- **Classes:** 15,587 unique individual whale/dolphin IDs
- **Species:** Whales (humpback, blue, fin, right, etc.) and dolphins
- **License:** CC-BY-NC-4.0 (see LICENSE_DATA.md)

## Model Architectures & Performance

| Architecture | Precision | Inference Time | Parameters |
|---|---|---|---|
| **Vision Transformer L/32** | **93%** | ~3.5s | ~307M |
| Vision Transformer B/16 | 91% | ~2.0s | ~86M |
| EfficientNet-B5 | 91% | ~1.8s | ~30M |
| EfficientNet-B0 | 88% | ~1.0s | ~5M |
| Swin Transformer | 90% | ~2.2s | ~29M |
| ResNet-101 | 85% | ~1.2s | ~45M |
| ResNet-54 | 82% | ~0.8s | ~26M |

**Production model:** Vision Transformer L/32 (best accuracy)

## Metrics

- **Precision:** >= 80% for clear 1920x1080 images (target met: 93%)
- **Sensitivity/Recall:** > 85% (target met)
- **Specificity:** > 90% (target met)
- **F1-score:** > 0.6 (target met)
- **Processing Speed:** < 8 seconds per 1920x1080 image (target met: ~3.5s)
- **Robustness:** Accuracy drop <= 20% on noisy images (target met)

## Input/Output

### Input

- **Format:** RGB images (JPEG, PNG)
- **Resolution:** Any (resized to 448x448 internally)
- **Normalization:** ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Output

- **Classification:** Individual whale ID (one of 15,587 classes)
- **Species mapping:** ID mapped to species name via config.yaml
- **Confidence:** Softmax probability (0.0-1.0)
- **Background removal:** Optional base64 PNG mask via rembg

## Training Configuration

- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-6)
- **Scheduler:** CosineAnnealingLR (T_max=500, min_lr=1e-6)
- **Loss:** ArcFace (s=30.0, m=0.50) + CrossEntropy
- **Batch Size:** 32 (train), 64 (valid)
- **Image Size:** 448x448
- **Augmentations:** ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast
- **Epochs:** 15 (best checkpoint: epoch 15)
- **Seed:** 2022 (fully reproducible)

## Limitations

- **Clear imagery required:** Performance degrades on heavily occluded, underwater, or very low-resolution images
- **Known species only:** Cannot identify species not present in the training dataset
- **Single-animal focus:** Best performance on images containing a single marine mammal
- **Lighting conditions:** Extreme backlighting or glare can reduce accuracy up to 20%
- **Geographic bias:** Training data predominantly from Northern Hemisphere whale populations

## Ethical Considerations

- **Conservation purpose:** Designed to support marine mammal conservation efforts
- **Data privacy:** No personally identifiable human data in training
- **Dual use:** Intended for scientific and conservation use only, not for commercial exploitation of marine resources
- **Bias:** Under-represented species may have lower identification accuracy

## Citation

```bibtex
@software{ecomarineai2024,
  title={EcoMarineAI: AI Library for Marine Mammal Identification},
  author={Baltsat, K.I. and Tarasov, A.A. and Vandanov, S.A. and Serov, A.I.},
  year={2024},
  url={https://github.com/baltsat/whales-identification}
}
```
