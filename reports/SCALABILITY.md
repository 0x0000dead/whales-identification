# Scalability benchmark

Measures wall-clock latency of the EcoMarineAI pipeline on batches of
N images drawn from `data/test_split/positives/`.

## Results

| N images | Total (s) | Per image (ms) |
|---------:|----------:|---------------:|
| 10 | 3.991 | 399 |
| 25 | 10.9934 | 440 |
| 50 | 23.0819 | 462 |
| 100 | 47.2903 | 473 |

## Linear regression

- slope: **0.482 s/image** (marginal cost)
- intercept: -0.9513 s (one-off warmup)
- R²: **1.0** (1.0 = perfect linear fit)

If R² ≥ 0.99 the pipeline has linear time complexity, as required
by the ТЗ (Параметр 3 — Масштабируемость системы).
