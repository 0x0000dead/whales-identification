# EcoMarineAI — Solution overview

**Audience:** project sponsors, grant reviewers, environmental-agency stakeholders, marine biologists without ML background.

## What is EcoMarineAI?

EcoMarineAI is an **open-source AI library and web service** that identifies individual whales and dolphins from aerial photographs. Upload a photo → get back the species, individual ID (when the animal is in the training set), confidence score, and — critically — an answer to "is this even a cetacean at all?".

It is designed to close three concrete gaps in current marine-mammal monitoring:

1. **Time cost.** Manual identification of individual whales from aerial surveys takes trained biologists minutes per frame. A drone flight generates thousands of frames.
2. **Consistency.** Different labellers disagree on hard cases. A single model applied uniformly gives reproducible results.
3. **Error visibility.** Traditional identification pipelines silently return *something* even on an image that isn't a whale at all. EcoMarineAI's **anti-fraud gate** explicitly rejects non-cetacean inputs with a documented reason.

## Who uses it?

| User                           | How they interact                          | Value                                                           |
|--------------------------------|--------------------------------------------|------------------------------------------------------------------|
| Marine biologist (field)       | Web UI or `whales-cli predict`             | Drop a photo, get species + confidence in < 1 second            |
| Research lab                   | Python CLI batch mode + CSV/SQL exports    | Process thousands of images into a searchable database          |
| Conservation NGO               | REST API `/v1/predict-single`              | Integrate predictions into a dashboard or mobile survey app     |
| Government monitoring agency   | `/metrics` Prometheus + `/v1/drift-stats`  | Continuous oversight of system availability and model drift      |
| ML researcher                  | Pretrained checkpoint on Hugging Face      | Fine-tune on their own species, publish derivative work          |
| Citizen scientist              | Web UI                                     | Contribute photos, learn about cetaceans                        |

## What it does (user-facing)

1. **Accepts** any RGB image (JPEG/PNG/WEBP/BMP) or a ZIP archive of many images.
2. **Filters** out anything that isn't a photo of a whale or dolphin (CLIP zero-shot anti-fraud gate, TNR ≥ 90.2%).
3. **Identifies** the individual animal from 13 837 known cetacean individuals across 30 species.
4. **Returns** a structured response with species name, individual ID, confidence, and an optional background-removed mask.
5. **Logs** every prediction to an in-memory drift monitor so operators see degradation before users complain.
6. **Exports** results to CSV, SQLite, or PostgreSQL via the `integrations/` scripts.

## What's under the hood

Two stages running in series:

1. **Anti-fraud gate**: OpenCLIP ViT-B/32 trained on LAION-2B. Zero-shot classification against 10 positive prompts (whale / dolphin / cetacean descriptors) and 14 negative prompts (text / buildings / fish / sharks / landscapes). Calibrated threshold chosen to give TNR ≥ 0.90 while preserving TPR ≥ 0.85.
2. **Identification model**: EfficientNet-B4 backbone + ArcFace head on 13 837 individuals. Cosine similarity between the image embedding and the class centroids gives a naturally interpretable confidence score.

Full technical details in [ML_ARCHITECTURE.md](ML_ARCHITECTURE.md).

## Measured performance

All numbers below are computed by `scripts/compute_metrics.py` on a reproducible in-repo test split (**100 positives** from Happy Whale + **102 negatives** from Intel Image Dataset, total **202 images**).

| Metric                 | Measured       | ТЗ target  | Status |
|------------------------|---------------:|-----------:|:------:|
| Sensitivity / TPR      | **0.9500**     | > 0.85     | ✓ |
| Specificity / TNR      | **0.9020**     | > 0.90     | ✓ |
| Precision              | **0.9048**     | ≥ 0.80     | ✓ |
| F1                     | **0.9268**     | > 0.60     | ✓ |
| Latency (p95, CPU)     | **519 ms**     | ≤ 8 000 ms | ✓ |
| Linear time complexity | **R² = 1.000** | linear     | ✓ |
| Noise robustness       | **0.0 % drop** | ≤ 20 %     | ✓ |
| Identified individuals | **13 837**     | ≥ 1 000    | ✓ |

## Why it matters beyond this one project

- **Open data × open models.** Code is MIT, models inherit CC-BY-NC-4.0 from the upstream Happy Whale dataset, everything is reproducible from Kaggle + HuggingFace mirrors.
- **Scientific rigour.** Every number in this document comes from a script that any reviewer can re-run on their own laptop in under a minute.
- **Extensibility.** Adding a new species means adding rows to the training CSV and re-fitting the ArcFace head — the rest of the pipeline doesn't change. Adding a new export integration is ~80 lines of Python (see `integrations/sqlite_sink.py`).
- **Failure visibility.** The CLIP anti-fraud gate makes the system honest about the edges of its knowledge. When you feed it a photo of your cat, it says so, loudly.

## What's next

See [ROADMAP.md](ROADMAP.md) for the detailed plan by ФСИ-grant milestone.
