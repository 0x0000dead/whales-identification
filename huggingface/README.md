---
license: cc-by-nc-4.0
license_link: LICENSE
library_name: pytorch
pipeline_tag: image-classification
tags:
  - pytorch
  - vision
  - image-classification
  - whale-identification
  - marine-mammals
  - conservation
  - vision-transformer
  - resnet
  - efficientnet
language:
  - en
  - ru
datasets:
  - happywhale
  - marine-mammals
widget:
  - src: https://github.com/0x0000dead/whales-identification/raw/main/data/sample/whale_sample.jpg
    example_title: Whale Identification
---

# EcoMarineAI: Whale and Dolphin Identification

Automated detection and identification of marine mammals (whales and dolphins) from aerial photography using deep learning.

## Model Description

This repository contains trained models for whale and dolphin individual identification from aerial photographs. The models use metric learning (ArcFace) to create embeddings that cluster similar whale individuals, enabling identification of 15,587+ unique individuals.

### Available Models

| Model                 | Architecture            | Precision | Speed | Best For      |
| --------------------- | ----------------------- | --------- | ----- | ------------- |
| `vit_l32_best.pth`    | Vision Transformer L/32 | 93%       | ~3.5s | Best accuracy |
| `resnet101.pth`       | ResNet-101              | 85%       | ~1.2s | Balanced      |
| `efficientnet-b5.pth` | EfficientNet-B5         | 91%       | ~1.8s | Good accuracy |

## Usage

### Installation

```bash
# Install huggingface_hub (use version 0.20.3 for huggingface-cli)
pip install huggingface_hub==0.20.3

# Download models
huggingface-cli download baltsat/Whales-Identification resnet101.pth --local-dir ./models
```

### Inference

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = torch.load("models/resnet101.pth", map_location="cpu")
model.eval()

# Preprocess image (448x448, ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("whale.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    predicted_id = output.argmax(dim=1).item()
```

## License

**Apache License 2.0 with Usage Restrictions**

Copyright (c) 2024 Baltsat Konstantin, Tarasov Artem, Vandanov Sergey, Serov Alexandr

### Important Usage Restrictions

**The trained models may NOT be used for commercial purposes** without explicit permission from the original data providers:

- **Happy Whale data:** CC-BY-NC-4.0 (non-commercial)
- **Ministry of Natural Resources RF data:** Government research-only terms
- **ImageNet pretrained weights:** Non-commercial research-only terms

See [LICENSE_MODELS.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_MODELS.md) for full details.

### Permitted Uses

- Academic research projects
- Educational purposes
- Non-profit conservation efforts
- Personal and non-commercial use
- Government marine mammal monitoring

### Prohibited Uses

- Commercial exploitation without data provider consent
- Applications that harm marine mammals
- Surveillance for hunting purposes

## Training Data

Models were trained on:

- **Happy Whale dataset** (https://happywhale.com, CC-BY-NC-4.0)
- **Ministry of Natural Resources of the Russian Federation** data
- ~60,000 training images of whales and dolphins

## Citation

```bibtex
@misc{whales-identification-2024,
  author = {Baltsat, Konstantin and Tarasov, Artem and Vandanov, Sergey and Serov, Alexandr},
  title = {EcoMarineAI: Automated Whale and Dolphin Identification from Aerial Photography},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/0x0000dead/whales-identification}
}
```

## Links

- **GitHub Repository:** https://github.com/0x0000dead/whales-identification
- **Documentation:** https://github.com/0x0000dead/whales-identification/wiki
- **Issues:** https://github.com/0x0000dead/whales-identification/issues

## Model Card

See [MODEL_CARD.md](https://github.com/0x0000dead/whales-identification/blob/main/MODEL_CARD.md) for detailed performance metrics, training specifications, and evaluation results.
