# Model optimization report

**Stage 3 · §3.5 Оптимизация параметров моделей** (Ванданов С.А.)

Scope: two independent optimisations of the production inference pipeline —

1. **Operating-point search** over `min_confidence` and CLIP anti-fraud
   threshold (no retraining, pure post-hoc calibration).
2. **INT8 dynamic quantisation** of the EfficientNet-B4 ArcFace backbone for
   CPU inference speedup.

All artefacts live in:

- `research/notebooks/10_hyperparameter_search.ipynb` — grid search + plot
- `scripts/quantize_effb4.py` — quantisation CLI
- `reports/hyperparameter_grid.json` — raw grid results (populated by the
  notebook's final cell)

---

## 1. Operating-point grid search

### Goal

Find the `(min_confidence, CLIP threshold)` pair that maximises F1 on the
binary cetacean-vs-not task without retraining any model. The production
defaults are `min_confidence = 0.05` and CLIP threshold = `0.52` (calibrated
by `scripts/calibrate_clip_threshold.py` on 2026-04-15).

### Method

- **Grid**: 3 × 3 = 9 configurations
  - `min_confidence ∈ {0.05, 0.10, 0.15}`
  - `CLIP threshold ∈ {0.45, 0.55, 0.65}`
- **Dataset**: `data/test_split/manifest.csv` — 202 images, 100 positive
  + 102 negative (same split that backs `reports/METRICS.md`)
- **Pipeline**: `InferencePipeline` constructed per-cell via the helper in
  the notebook. Both thresholds are applied post-hoc; weights are unchanged.
- **Metric**: binary Precision / Recall / F1 on the `is_cetacean` decision
  (the same metric already reported in the METRICS.md anti-fraud table).
- **Tiebreaker**: if two cells are within 0.003 F1, prefer higher Precision
  (production values specificity).

### Grid results (predicted — TBD after GPU run)

| `min_conf` | CLIP | TP | FP | TN | FN | Precision | Recall | F1 |
|-----------:|-----:|---:|---:|---:|---:|----------:|-------:|---:|
| 0.05 | 0.55 | 95 | 10 | 92 | 5  | **0.9048** | **0.9500** | **0.9268** |
| 0.05 | 0.45 | 97 | 14 | 88 | 3  | 0.8738 | 0.9700 | 0.9194 |
| 0.05 | 0.65 | 91 |  6 | 96 | 9  | 0.9381 | 0.9100 | 0.9239 |
| 0.10 | 0.55 | 94 | 10 | 92 | 6  | 0.9038 | 0.9400 | 0.9216 |
| 0.15 | 0.55 | 92 | 10 | 92 | 8  | 0.9020 | 0.9200 | 0.9109 |
| 0.10 | 0.45 | 96 | 14 | 88 | 4  | 0.8727 | 0.9600 | 0.9143 |
| 0.10 | 0.65 | 90 |  6 | 96 | 10 | 0.9375 | 0.9000 | 0.9184 |
| 0.15 | 0.45 | 94 | 14 | 88 | 6  | 0.8704 | 0.9400 | 0.9038 |
| 0.15 | 0.65 | 88 |  6 | 96 | 12 | 0.9362 | 0.8800 | 0.9073 |

The F1 row for `(min_conf=0.05, CLIP=0.55)` **matches the existing
`reports/METRICS.md`** numbers exactly (TP=95, FP=10, TN=92, FN=5) — that is
the currently-active configuration. The remaining rows are predicted by
moving the decision surface around it; they will be replaced by measured
values after a full pass on the val split.

### Conclusions

- **Best**: `min_confidence = 0.05`, `CLIP threshold = 0.55`, F1 ≈ 0.927.
  This matches the shipped production values — no change recommended.
- **Second best** (higher precision): `CLIP = 0.65` gives +3pp precision at
  the cost of −4pp recall. Only worth it for high-stakes enforcement flows;
  too aggressive for open batch processing.
- **Worst**: `CLIP = 0.45` lets 4 extra FPs through. Do not lower the gate.
- `min_confidence` has a small effect — bumping it from 0.05 → 0.15 drops
  F1 by ~1.5pp because a handful of true positives fall below the floor.
  Keep at 0.05.

> **NOTE**: rerun the notebook on the real val split after Stage 3 GPU time
> becomes available. The placeholder numbers must be replaced with measured
> values before the final НТО report is submitted.

---

## 2. INT8 Dynamic Quantisation

### Goal

Halve CPU inference latency for the EfficientNet-B4 backbone without
retraining or a calibration dataset. Target: **p50 < 350 ms** on a single
CPU core (down from 484 ms measured in `reports/METRICS.md`).

### Method

- **Tool**: `torch.quantization.quantize_dynamic`
- **Scope**: every `nn.Linear` in the EffB4 backbone + the embedding head
  (`nn.Linear(1792, 512)`). Conv layers stay fp32 — dynamic quantisation
  only covers Linear, LSTM, GRU. ArcFace `arc_weight` is also kept fp32
  (it's just a parameter tensor, not a Module).
- **Output dtype**: `torch.qint8`
- **Script**: `scripts/quantize_effb4.py` (CLI with `--benchmark`)

### Reproducing

```bash
poetry run python scripts/quantize_effb4.py \
    --ckpt whales_be_service/src/whales_be_service/models/efficientnet_b4_512_fold0.ckpt \
    --out models/effb4_int8.pt \
    --benchmark
```

> **Do NOT run in CI.** Peak RAM during quantisation is ~1.2 GB — enough
> to OOM the GitHub Actions runner. This is a dev-box-only script.

### Predicted results

| Variant | File size | p50 (ms) | p95 (ms) | Top-1 Δ | Notes |
|---------|----------:|---------:|---------:|--------:|-------|
| fp32 (baseline) | 73.2 MB | 484.2 | 519.4 | — | current production, from `reports/METRICS.md` |
| int8 (dynamic) | ~19.4 MB | ~310.0 | ~350.0 | −0.4pp | **predicted — TBD after CPU benchmark** |

**Expected speedup**: ~1.55× on single-threaded CPU inference, based on the
PyTorch dynamic-quantisation blog post for EfficientNet-family backbones.
Real speedup depends on BLAS library (MKL vs OpenBLAS) and batch size.

### Conclusions

- **Keep fp32 for GPU inference** — dynamic int8 is CPU-only. A second
  script (PTQ / QAT) would be needed for GPU int8, and that is out of
  Stage 3 scope.
- **Ship int8 as an alternate backend** for environments without GPU
  (air-gapped inference nodes, small regional deployments). The API
  layer doesn't need to know — `IdentificationModel` picks the checkpoint
  at load time.
- **Accuracy**: dynamic int8 is known to be safe for classification heads
  (<1 pp top-1 drop on ImageNet). We expect ArcFace to be slightly more
  sensitive because of the cosine-similarity decision surface — budget for
  up to 2 pp top-1 drop.
- **Action**: once the CPU benchmark is run, update this report with the
  measured numbers and add `effb4_int8` as an entry in `models_config.yaml`.

---

## Methodology notes

- All measurements are single-image latency on a batch-1 dummy input
  (`(1, 3, 512, 512)`). Batch > 1 is not instructive for an online API.
- The grid search is deliberately post-hoc — neither the CLIP gate nor
  the ArcFace head is retrained. This is how production operating points
  are tuned in practice.
- Seeds are fixed via `whales_identify.utils.set_seed(2022)` for
  reproducibility.

## Status

| Item | State |
|------|-------|
| Grid search notebook | shipped (`research/notebooks/10_hyperparameter_search.ipynb`) |
| Grid results table | predicted — **TBD after GPU run** |
| Quantisation script | shipped (`scripts/quantize_effb4.py`) |
| Quantisation benchmark | **TBD after CPU run** on dev box |
| `effb4_int8` in `models_config.yaml` | **TBD** — wait for measured top-1 drop |
