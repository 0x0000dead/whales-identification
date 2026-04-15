# EcoMarineAI Metrics Report

_Generated: 2026-04-15T11:33:06.977796+00:00_
_Manifest: `data/test_split/manifest.csv`_
_Sample size: 60_
_Model version: `vit_l32-v1`_

## Anti-fraud (CLIP gate, binary)

| Metric                       | Value     |
|------------------------------|-----------|
| Positives                    | 30 |
| Negatives                    | 30 |
| TP / FP / TN / FN            | 29 / 2 / 28 / 1 |
| **TPR / Sensitivity / Recall** | **0.9667** |
| **TNR / Specificity**        | **0.9333** |
| Precision                    | 0.9355 |
| F1                           | 0.9508 |
| ROC-AUC (cetacean_score)     | 0.9922 |

## Identification (multiclass, on positives only)

| Metric                       | Value     |
|------------------------------|-----------|
| Samples                      | 30 |
| Unique individuals           | 30 |
| Top-1 accuracy               | 0.1667 |

## Performance

| Metric                       | Value     |
|------------------------------|-----------|
| Samples timed                | 60 |
| Latency p50 / p95 / p99 (ms) | 489.27 / 540.31 / 607.56 |
| Latency mean (ms)            | 278.81 |
