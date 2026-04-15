# Scalability benchmark

Measures wall-clock latency of the EcoMarineAI pipeline on batches of
N images drawn from `data/test_split/positives/`.

## Results

| N images | Total (s) | Per image (ms) |
|---------:|----------:|---------------:|
| 5 | 2.5497 | 510 |
| 10 | 5.1292 | 513 |
| 20 | 10.3924 | 520 |
| 30 | 14.8035 | 493 |

## Linear regression

- slope: **0.493 s/image** (marginal cost)
- intercept: 0.2081 s (one-off warmup)
- R²: **0.9982** (1.0 = perfect linear fit)

If R² ≥ 0.99 the pipeline has linear time complexity, as required
by the ТЗ (Параметр 3 — Масштабируемость системы).
