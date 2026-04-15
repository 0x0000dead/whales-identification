# EcoMarineAI Metrics Report

_Generated: 2026-04-15T13:15:39.392866+00:00_
_Manifest: `data/test_split/manifest.csv`_
_Sample size: 202_
_Model version: `effb4-arcface-v1`_

## Anti-fraud (CLIP gate, binary)

| Metric                       | Value     |
|------------------------------|-----------|
| Positives                    | 100 |
| Negatives                    | 102 |
| TP / FP / TN / FN            | 95 / 10 / 92 / 5 |
| **TPR / Sensitivity / Recall** | **0.95** |
| **TNR / Specificity**        | **0.902** |
| Precision                    | 0.9048 |
| F1                           | 0.9268 |
| ROC-AUC (cetacean_score)     | 0.984 |

## Identification (multiclass, on positives only)

| Metric                       | Value     |
|------------------------------|-----------|
| Samples                      | 100 |
| Unique individuals           | 93 |
| Top-1 accuracy               | 0.22 |
| Top-5 accuracy               | 0.25 |

## Image clarity (ТЗ §Параметр 1, Laplacian variance)

The ТЗ defines «sufficiently clear» as Laplacian variance within 5%% of
the dataset mean. We compute the variance per image and list how many
pass the threshold.

| Metric                       | Value     |
|------------------------------|-----------|
| Mean Laplacian variance      | 4485.01 |
| Min / Max                    | 4.96 / 40416.64 |
| ТЗ threshold (mean × 0.95)   | 4260.76 |
| Images above threshold       | 77 |
| Images below threshold       | 125 |

## Performance

| Metric                       | Value     |
|------------------------------|-----------|
| Samples timed                | 202 |
| Latency p50 / p95 / p99 (ms) | 484.15 / 519.42 / 597.24 |
| Latency mean (ms)            | 277.33 |
