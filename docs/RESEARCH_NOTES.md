# Research notes

This document captures the "why" behind the technical choices in EcoMarineAI: which alternatives we considered, what we tried, and why the final architecture won out.

---

## 1. Why ArcFace + cosine classification?

### Problem shape

Individual whale identification is an **open-set retrieval** task:

- The training set has ~15 k individuals but new animals appear every season.
- Different photos of the same individual vary in pose, lighting, partial occlusion.
- Two different animals of the same species look nearly identical to a non-expert.

Classic softmax classification treats unseen individuals as "noise", while ArcFace (Deng et al., CVPR 2019) explicitly optimises for an **angular margin** on the hypersphere, which:

1. Pulls embeddings of the same class together.
2. Pushes embeddings of different classes apart.
3. Makes the resulting features **directly usable for nearest-neighbour retrieval**.

Reference: _Deng, J. et al. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition". CVPR._

### Alternatives considered

| Method                          | Why rejected                                               |
|---------------------------------|------------------------------------------------------------|
| Softmax cross-entropy           | No angular margin → embeddings collapse, poor open-set    |
| Triplet loss                    | Hard-negative mining is fiddly; ArcFace is easier to tune  |
| CosFace (Wang et al. 2018)      | Very similar to ArcFace; ArcFace slightly outperforms on whale competitions |
| SphereFace (Liu et al. 2017)    | Multiplicative margin is unstable in practice              |
| CircleLoss                      | Newer (2020) — not enough published Happy Whale experiments |
| Pure metric learning (Siamese)  | Requires explicit pair mining, slow                       |

Our model uses `s=30.0, m=0.50, easy_margin=False`, which are the stock Happy Whale competition defaults.

---

## 2. Why OpenCLIP for the anti-fraud gate?

### Problem shape

At inference time we have **no labels** for "is this a cetacean photo or not". The user just uploads an image. We want to reject non-cetacean content before wasting compute on identification.

Three families of solutions:

1. **Train a binary classifier** (cetacean vs. random). Needs ~10 k labelled negatives and a training run for every new deployment.
2. **Out-of-distribution detection** on the identification model itself — e.g. maximum softmax probability, energy score, Mahalanobis distance. Doesn't require new data but is noisy and overconfident.
3. **Zero-shot vision-language classifier** — CLIP. Compare the image against text prompts like "a photo of a whale" vs "a photo of text". No training data, no fine-tuning.

We chose (3) because:

- **Zero training data.** The model ships with the code.
- **Robust to distribution shift.** CLIP on LAION-2B has seen enormous amounts of imagery.
- **Easy to tune.** Rejected species? Add a new prompt. No retraining.
- **Cheap.** ViT-B/32 is ~150 MB and ~25 ms per image on GPU.

Reference: _Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision". ICML._

### Why OpenCLIP (laion2b_s34b_b79k) rather than OpenAI CLIP?

- **Bigger training set.** LAION-2B is 10× OpenAI's dataset.
- **Permissive licence.** LAION-2B checkpoints are Apache 2.0; OpenAI CLIP weights have restrictions.
- **Better zero-shot accuracy.** On animal-class ImageNet subsets, LAION-2B beats the OpenAI version by 2–5 percentage points (see `open_clip_torch` model cards).
- **Same API.** Drop-in replacement.

### Prompt engineering notes

We use 10 positive + 14 negative prompts. The ratio matters: too few negatives and specificity suffers; too many and sensitivity drops. The negative set was chosen to cover the adversarial cases we saw during expert review:

- `"a photo of a fish"` — important because cetaceans are mammals, not fish; CLIP's prior from LAION can confuse them if we don't explicitly separate.
- `"a photo of a shark"` — visually similar silhouette.
- `"a photo of text on a blank page"` — common adversarial input (screenshot of a document).
- `"a photo of a boat on water"` — typical "wildlife photography at sea" false positive.

The calibration script (`scripts/calibrate_clip_threshold.py`) sweeps thresholds on the real test split and picks the smallest one satisfying TNR ≥ 0.90 / TPR ≥ 0.85. Current calibration: **threshold = 0.30**, achieving TPR = 0.9667 / TNR = 0.9333.

---

## 3. Why EfficientNet-B4 rather than ViT-L or Swin?

Table from our internal benchmark (and consistent with Happy Whale competition leaderboards):

| Backbone              | Params | Train time (1 fold) | Inference (CPU) | Val top-5 |
|-----------------------|-------:|--------------------:|----------------:|----------:|
| ResNet-50             |   25 M |               ~4 h  |        ~200 ms  |      0.62 |
| ResNet-101            |   44 M |               ~6 h  |        ~350 ms  |      0.65 |
| EfficientNet-B0       |    5 M |               ~3 h  |        ~200 ms  |      0.66 |
| **EfficientNet-B4**   | **19 M**|              **~5 h**|       **~500 ms**|   **0.74**|
| EfficientNet-B5       |   30 M |               ~8 h  |        ~700 ms  |      0.75 |
| EfficientNet-B7       |   66 M |              ~12 h  |       ~1 200 ms |      0.76 |
| ViT-B/16              |   86 M |              ~10 h  |        ~800 ms  |      0.73 |
| ViT-L/32              |  307 M |              ~20 h  |       ~3 500 ms |      0.75 |
| Swin-T                |   29 M |               ~8 h  |        ~500 ms  |      0.72 |

EfficientNet-B4 is the **knee of the price/quality curve**. B5 and B7 gain ~1 pp of accuracy for ~40% and ~140% more cost respectively.

ViT-L/32 was our first attempt (see `research/notebooks/02_*`) — the architecture matches the ТЗ mention of Vision Transformers, but training from scratch on a TPU-less budget pushed it toward EffB4 + metric learning as the production choice.

---

## 4. What didn't work

A record of alternatives we tried, so future contributors don't repeat the mistakes:

### 4.1 Random augmentation during inference (TTA)

Standard at train time, but at inference TTA added ~400 ms for only +0.5 pp top-1. Not worth it for CPU-only deployments.

### 4.2 ImageNet pre-training → fine-tune on 30 species

We tried a pure species-classification head (30 classes). Worked fine on closed-set species assignment but couldn't handle **individual** identification. Reverted to ArcFace on individual_id.

### 4.3 Background removal before identification

Using `rembg` to strip backgrounds gave a small boost on humpback whales (clear silhouette) but hurt dolphins (noisy masks in water). Kept `rembg` as an optional response field only.

### 4.4 Single-stage detection + classification (Faster R-CNN)

Requires bounding-box annotations, of which we only have 5 201 (`data/backfin_annotations.csv`). Training a detector on that subset gave poor generalisation. Parked until we have more data.

### 4.5 Larger CLIP (ViT-L/14)

ViT-L/14 gave +0.3% TNR but made the gate 4× slower. Not worth it.

### 4.6 Hard-coded noise augmentation in the gate

Idea: add Gaussian noise to negative prompts' text embeddings to make the gate more robust. Didn't help; the current calibration threshold already handles noise well enough (6.9% drop — see `reports/NOISE_ROBUSTNESS.md`).

---

## 5. Future research directions

1. **Self-supervised pretraining on whale videos.** SimCLR / DINO on raw drone footage → better backbones than ImageNet.
2. **FAISS-backed nearest-neighbour retrieval.** Pre-compute all training embeddings, use ArcFace features as a retrieval index. Enables top-K ranking and open-set handling.
3. **Test-time ensembling over folds.** Run 5 × EfficientNet-B4 fold models, average probabilities. Expected +1–2 pp top-1.
4. **Multimodal input fusion.** Combine the drone image with GPS + timestamp to narrow down the list of likely individuals (e.g. a whale seen yesterday in that bay).
5. **Temporal consistency.** For video streams, use a Kalman filter to smooth predictions over consecutive frames.
6. **Active learning.** Low-confidence predictions go into a human-review queue; labelled corrections update the model monthly.
7. **Cross-species transfer.** The ArcFace backbone learned from whales may transfer to dolphins, beluga, and even non-cetacean marine mammals (seals) with minimal fine-tuning.

---

## 6. Related literature

- **Happy Whale competition reports** — https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion
- **MegaDetector** (Beery et al. 2019) — camera-trap wildlife detector; inspired our anti-fraud approach.
- **BIOSCAN-1M** (Gharaee et al. 2023) — insect identification at scale with metric learning.
- **iNaturalist** — https://www.inaturalist.org/ — community-driven species ID, validated backbone choices.
- **Darwin Core** — https://dwc.tdwg.org/terms/ — structured output format for biodiversity observations.
- **GBIF** — https://www.gbif.org/ — global biodiversity data aggregator.

## 7. Honest limitations

1. **The training set has class imbalance.** Some species have thousands of images; others have a handful. Per-species precision varies substantially.
2. **Geographic bias.** Happy Whale data is overwhelmingly North Atlantic and North Pacific. Southern Ocean species (e.g., Antarctic minke whale) are under-represented.
3. **Photo quality bias.** Most training images are from professional drone operators. Grainy smartphone photos from shore-based observers may not match the distribution.
4. **Labelling noise.** The Happy Whale individual_id column has known mislabelled examples (e.g. `bottlenose_dolpin` typo). We preserve the labels as-is for reproducibility; a future release may denoise.
5. **No uncertainty quantification.** We report top-1 probability but not a confidence interval. For risk-sensitive downstream decisions (population estimates, conservation decisions), a Bayesian head would be preferable.
