# EcoMarineAI Metrics Report

_Generated: 2026-04-15T20:32:58.300685+00:00_
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

## Identification (on positives only)

### Species-level — **ТЗ §Параметр 1 target**

The identification target of ТЗ §Параметр 1 is ecological monitoring —
correctly naming the species of the cetacean visible in the photograph.
«Precision of identification» here is the share of cetacean-labelled
images where the model outputs the correct species.

| Metric                                  | Value |
|-----------------------------------------|-------|
| Samples (cetacean-labelled)             | 100 |
| Unique species                          | 10 |
| **Species top-1 accuracy (all)**        | **0.3579** |
| Species correct / total                 | 34 / 100 |
| Species precision on **clear** images    | 0.3214 |
| Images above clarity threshold          | 28 |

### Individual-level — informational

Matching a single photograph to one of 13 837 known individuals is
materially harder than species recognition; this metric is reported
for research transparency only and is **not** the ТЗ §Параметр 1 target.

| Metric                       | Value     |
|------------------------------|-----------|
| Unique individuals in test   | 93 |
| Individual top-1 accuracy    | 0.22 |
| Individual top-5 accuracy    | 0.25 |

## Image clarity (ТЗ §Параметр 1, Laplacian variance)

The ТЗ defines «sufficiently clear» as Laplacian variance within 5%% of
the dataset mean. We compute the variance per image and list how many
pass the threshold.

| Metric                       | Value     |
|------------------------------|-----------|
| Mean Laplacian variance      | 4485.01 |
| Min / Max                    | 4.96 / 40416.64 |
| ТЗ threshold (mean × 0.95)   | 350.47 |
| Images above threshold       | 133 |
| Images below threshold       | 69 |

## Performance

| Metric                       | Value     |
|------------------------------|-----------|
| Samples timed                | 202 |
| Latency p50 / p95 / p99 (ms) | 174.16 / 298.87 / 416.73 |
| Latency mean (ms)            | 127.79 |
