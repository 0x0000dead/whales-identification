# Model License

## Apache License 2.0 with Usage Restrictions

Copyright (c) 2024 Baltsat Konstantin, Tarasov Artem, Vandanov Sergey, Serov Alexandr

Licensed under the Apache License, Version 2.0 (the "License");
you may not use these models except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

## Important Usage Restrictions

**⚠️ COMMERCIAL USE RESTRICTIONS**

The trained models in this repository were developed using datasets that include data licensed under **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International). As a result:

1. **The trained models may NOT be used for commercial purposes** without explicit permission from the original data providers (HappyWhale and Ministry of Natural Resources of the Russian Federation).

2. This restriction applies specifically to the **trained model weights** (`.pt`, `.pth`, `.onnx` files) and any derivatives thereof.

3. Under EU law, models trained on non-commercial data inherit those restrictions when the model is sold or used commercially.

4. **Pretrained Models (Transfer Learning):** This project uses pretrained models (ResNet, EfficientNet, ViT, Swin Transformer) that were originally trained on ImageNet. ImageNet has **non-commercial research-only terms**, adding another layer of restriction beyond HappyWhale and Ministry RF data.

---

## Pretrained Model Foundations

**⚠️ CRITICAL: ImageNet Pretrained Weights Restrictions**

Our fine-tuned models are built upon pretrained models from the following sources, all of which use ImageNet for initial training:

| Pretrained Model | Source | Code License | Pretrained Weights | ImageNet Terms |
|------------------|--------|--------------|-------------------|----------------|
| **ResNet-50/101** | [torchvision](https://github.com/pytorch/vision) | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | [ImageNet-1k](https://www.image-net.org/download.php) | Non-commercial |
| **EfficientNet-B0/B5** | [TIMM](https://github.com/huggingface/pytorch-image-models) (Google) | [Apache 2.0](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) | [ImageNet-1k](https://www.image-net.org/download.php) | Non-commercial |
| **`tf_efficientnet_b0_ns`** | [Google Noisy Student](https://github.com/google-research/noisystudent) | [Apache 2.0](https://github.com/google-research/noisystudent/blob/master/LICENSE) | ImageNet + JFT-300M | Non-commercial |
| **ViT-B/16, ViT-L/32** | [Google ViT](https://github.com/google-research/vision_transformer) | [Apache 2.0](https://github.com/google-research/vision_transformer/blob/main/LICENSE) | [ImageNet-21k](https://www.image-net.org/download.php) | Non-commercial |
| **Swin-T, Swin-L** | [Microsoft Swin](https://github.com/microsoft/Swin-Transformer) | [MIT](https://github.com/microsoft/Swin-Transformer/blob/main/LICENSE) | [ImageNet-22k](https://www.image-net.org/download.php) | Non-commercial |
| **ConvNeXt-L** | [Facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | [CC-BY-NC 4.0](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE) | [ImageNet-22k](https://www.image-net.org/download.php) | Non-commercial |

### ImageNet Licensing Summary

**ImageNet Dataset Terms:**
- **Official Website:** https://www.image-net.org/
- **Download & Terms:** https://www.image-net.org/download.php
- **Terms of Access:** https://image-net.org/download-images.php
- ImageNet is released for **non-commercial research and educational purposes only**
- Any model trained on ImageNet (or fine-tuned from ImageNet pretrained weights) inherits these restrictions
- Commercial use of ImageNet-derived models requires explicit licensing from ImageNet organizers
- **Contact for commercial use:** feedback@image-net.org

**Implications for This Project:**
1. Even though the pretrained model **code** uses permissive licenses (Apache 2.0, BSD)
2. The pretrained **weights** are restricted by ImageNet terms
3. Our fine-tuned models are derived from these ImageNet weights
4. Therefore: **Commercial use is prohibited**

### Combined Restrictions

Our models face **triple restrictions** on commercial use:
1. **HappyWhale Training Data:** CC-BY-NC-4.0 (non-commercial)
2. **Ministry RF Training Data:** Government research-only terms
3. **ImageNet Pretrained Weights:** Non-commercial research-only terms

**Any ONE of these restrictions is sufficient to prohibit commercial use. All three apply simultaneously.**

### Pretrained Model Attribution

When using our models, you must also acknowledge the pretrained model sources:

```
Pretrained Model Attributions:
- ResNet: torchvision (BSD-3-Clause), https://github.com/pytorch/vision, trained on ImageNet-1k
- EfficientNet: TIMM/Google (Apache 2.0), https://github.com/huggingface/pytorch-image-models, trained on ImageNet-1k
- Vision Transformer: Google Research (Apache 2.0), https://github.com/google-research/vision_transformer, trained on ImageNet-21k
- Swin Transformer: Microsoft (MIT), https://github.com/microsoft/Swin-Transformer, trained on ImageNet-22k
- ConvNeXt: Facebook/Meta (CC-BY-NC 4.0), https://github.com/facebookresearch/ConvNeXt, trained on ImageNet-22k

All pretrained weights subject to ImageNet non-commercial terms (https://www.image-net.org/download.php).
```

---

## Permitted Uses

✅ **Research and Educational Use**
- Academic research projects
- Educational purposes in universities and research institutions
- Non-profit conservation efforts
- Scientific publications and presentations

✅ **Personal and Non-Commercial Use**
- Wildlife observation and documentation
- Personal hobby projects
- Open-source collaboration for research purposes

✅ **Government and Conservation Organizations**
- Marine mammal monitoring and conservation
- Environmental protection initiatives
- Scientific surveys and population studies

---

## Models Covered by This License

The following trained models are subject to this license:

| Model Name | Architecture | Version | File Location |
|------------|--------------|---------|---------------|
| `model-e15.pt` | Vision Transformer L/32 | v1.0 (epoch 15) | `models/model-e15.pt` |
| `resnet101.pth` | ResNet-101 | v1.0 | `models/resnet101.pth` |
| `efficientnet-b5.pth` | EfficientNet-B5 | v1.0 | `models/efficientnet-b5.pth` |
| Other models | Various | - | `models/*.pt`, `models/*.pth` |

**Note:** ONNX-optimized models (`.onnx` files) are also subject to the same license terms.

---

## Model Storage and Distribution

The models are distributed through:
- **HuggingFace Hub:** [baltsat/Whales-Identification](https://huggingface.co/baltsat/Whales-Identification)
- **Yandex Disk:** [Public folder](https://disk.yandex.ru/d/GshqU9o6nNz7ZA)
- **GitHub Releases:** (for stable versions only)

Models are **NOT** stored directly in the GitHub repository due to size constraints (`.gitignore` exclusion).

---

## Attribution Requirements

When using these models, you must provide proper attribution:

```
@misc{whales-identification-2024,
  author = {Baltsat, Konstantin and Tarasov, Artem and Vandanov, Sergey and Serov, Alexandr},
  title = {EcoMarineAI: Automated Whale and Dolphin Identification from Aerial Photography},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/0x0000dead/whales-identification},
  note = {Models trained on HappyWhale and Ministry of Natural Resources RF data}
}
```

Additionally, you must acknowledge the original data sources:
- **HappyWhale:** "This work uses data from HappyWhale (happywhale.com), licensed under CC-BY-NC-4.0"
- **Ministry of Natural Resources RF:** "This work uses data provided by the Ministry of Natural Resources and Ecology of the Russian Federation"

---

## Model Limitations and Responsible Use

### Technical Limitations
- Models are optimized for aerial photography of marine mammals
- Performance degrades significantly (up to 20%) on noisy, blurry, or low-quality images
- Best results with clear dorsal fin or body pattern visibility
- Not suitable for underwater images or video streams
- Occlusions >50% significantly reduce accuracy

### Ethical Considerations
- **Privacy:** No human subjects are present in training data
- **Bias:** Models were primarily tested on orcas and humpback whales; performance on other species may vary
- **Environmental Impact:** Model training carbon footprint is estimated at ~XX kg CO2 (calculation in progress)
- **Conservation Impact:** Models are intended to aid marine mammal conservation efforts

### Prohibited Uses
The following uses are explicitly prohibited:

❌ Commercial exploitation without data provider consent
❌ Applications that harm marine mammals or their habitats
❌ Surveillance or tracking of marine mammals for hunting purposes
❌ Misrepresentation of model capabilities or accuracy
❌ Use in contexts that violate local wildlife protection laws

---

## Model Versioning and Updates

Model versions follow semantic versioning: `vMAJOR.MINOR.PATCH`

- **MAJOR:** Architectural changes, incompatible API changes
- **MINOR:** New features, backward-compatible improvements
- **PATCH:** Bug fixes, performance optimizations

**Current Stable Version:** v1.0 (January 2025)

**Model Card:** See [MODEL_CARD.md](MODEL_CARD.md) for detailed performance metrics, training data specifications, and evaluation results.

---

## Contact for Commercial Licensing

For inquiries regarding commercial use, custom licensing, or partnerships:

- **GitHub Issues:** https://github.com/0x0000dead/whales-identification/issues
- **Data Provider (HappyWhale):** support@happywhale.com
- **Data Provider (Ministry):** [Appropriate ministry contact]

---

## License Compatibility

### Compatible with:
- Apache 2.0 licensed code
- MIT licensed code
- BSD licensed code
- Other permissive licenses (for non-commercial use)

### NOT Compatible with:
- GPL licensed code (due to Apache 2.0 provisions)
- Commercial software (due to training data restrictions)
- Any license that requires commercial use rights

---

## Disclaimer

THE MODELS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODELS OR THE USE OR OTHER DEALINGS IN THE MODELS.

The models' predictions should not be the sole basis for critical conservation decisions. Always validate model outputs with expert marine biologist review.

---

## Updates to This License

This license may be updated to reflect changes in:
- Training data licensing terms
- Legal requirements
- Model capabilities and limitations

**Last Updated:** January 2025
**Version:** 1.0
**Effective Date:** January 2025
