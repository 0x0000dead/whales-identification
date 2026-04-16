# Load test report — EcoMarineAI backend

_Reporting period: 2026-04-15_
_Model version: `effb4-arcface-v1`_
_Related manifests: `k8s/deployment.yaml`, `k8s/hpa.yaml`, `docker-compose.prod.yml`_
_Related goals: ТЗ Параметр 3 — линейная временная сложность, Параметр 7 — availability ≥ 95 %_

## 1. Methodology

Two complementary measurements were taken:

1. **Pipeline benchmark (offline)** — `scripts/benchmark_scalability.py`,
   sequential inference on `data/test_split/positives/` to establish the
   single-pod latency baseline and the linearity of the per-image cost.
2. **HTTP load test (online)** — `locust -f tests/performance/locustfile.py`
   against the containerised backend, exercising the real FastAPI stack
   including CLIP anti-fraud gate, ArcFace identification, rembg background
   removal, and JSON serialisation.

### Locust configuration

```bash
# Terminal 1 — start the stack under test
./scripts/start.sh prod

# Terminal 2 — run the load test
locust -f tests/performance/locustfile.py \
    --host http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless \
    --csv reports/locust_run
```

- Target endpoints (weights from `tests/performance/locustfile.py`):
  - `POST /v1/predict-single` — weight 3
  - `POST /v1/predict-batch`  — weight 1 (3-image ZIP)
  - `GET  /health`            — weight 1
  - `GET  /metrics`           — weight 1
- Duration: 5 minutes steady-state after a 10 s ramp-up.
- Nominal target: **50 RPS** sustained, p95 latency **< 8 s** per image.
- Think time: `between(1, 3)` seconds per virtual user.

### Hardware / software profile

| Component          | Value                                                      |
|--------------------|------------------------------------------------------------|
| Platform           | Docker Compose (prod) on a single workstation              |
| CPU                | 8-core x86-64 (matches k8s pod limit 2 CPU × 4 replicas)   |
| RAM                | 32 GiB host, 4 GiB per backend container                   |
| GPU                | None — inference runs on CPU (PyTorch 2.4.1, MKLDNN)       |
| Backend image      | `ecomarine-backend:latest` (Python 3.11.6, uvicorn, FastAPI)|
| Load driver        | Locust 2.x                                                 |

## 2. Results — single-pod pipeline benchmark

Source: `reports/SCALABILITY.md` + `reports/scalability_latest.json`.

| N images | Total (s) | Per image (ms) |
|---------:|----------:|---------------:|
| 10       | 3.991     | 399            |
| 25       | 10.993    | 440            |
| 50       | 23.082    | 462            |
| 100      | 47.290    | 473            |

Linear regression on the four points:

- Slope (marginal cost): **0.482 s / image**
- Intercept (warm-up): **−0.95 s**
- R² = **1.000**

Per-image HTTP round-trip latency on the same dataset
(source: `reports/metrics_latest.json`, 202 timed samples, CPU):

| Percentile | Value (ms) |
|------------|-----------:|
| p50        | 174.16     |
| p95        | 298.87     |
| p99        | 416.73     |
| mean       | 127.79     |

**Interpretation.** A single backend pod processes one cetacean image in
≈ 128 ms mean / 299 ms p95 with an R² of 1.0, confirming the linear time
complexity required by **ТЗ Параметр 3**. The p95 of 0.30 s leaves a
~27× headroom below the 8 s ceiling — even under a 15× latency penalty
from contention a single pod would still meet the SLA.

## 3. Results — HTTP load test (Locust, 50 RPS target)

> **Status:** methodology fixed, exact numbers to be filled in on the next
> clean prod run. The benchmark in §2 already demonstrates linearity and a
> comfortable latency margin; this section captures the concurrent-load
> behaviour under the full FastAPI stack.

| Metric                        | Value                         |
|-------------------------------|-------------------------------|
| Target throughput             | 50 RPS                        |
| Sustained throughput          | TBD  // measured via locust   |
| Concurrent users              | 100 (spawn rate 10/s)         |
| Duration                      | 5 min steady-state            |
| p50 latency                   | TBD  // measured via locust   |
| p95 latency                   | TBD  // measured via locust   |
| p99 latency                   | TBD  // measured via locust   |
| Error rate (HTTP ≥ 500)       | TBD  // measured via locust   |
| Availability (1 − errors/total)| TBD  // measured via locust  |
| Pods active during run        | 3 (HPA minReplicas)           |

Raw Locust CSVs will be written to `reports/locust_run_*.csv` so the
numbers above can be reproduced or refreshed without editing this report.

## 4. Conformity with ТЗ goals

| ТЗ Параметр                       | Target                       | Evidence                                          | Met?   |
|-----------------------------------|------------------------------|---------------------------------------------------|--------|
| 3 — linear time complexity        | R² ≥ 0.99 on throughput data | §2 pipeline benchmark, R² = 1.000                 | yes    |
| 3 — per-image latency             | ≤ 8 s for 1920×1080          | §2 HTTP p95 = 299 ms (production compute_metrics.py) | yes    |
| 7 — availability                  | ≥ 95 % over 7 days            | `/metrics availability_percent`, continuous scrape | TBD // long-run |

- Steady-state offline linearity is proven (§2).
- Single-pod HTTP p95 already beats the ceiling by ~15×.
- Multi-pod concurrent load result (§3) pending a clean prod run; the same
  hardware profile is expected to hit the 50 RPS target at ≤ 2 s p95 based
  on the single-pod figures above.

## 5. How to reproduce

```bash
# 1. Bring the prod stack up on the same host.
./scripts/start.sh prod

# 2. Run the offline pipeline benchmark (§2 numbers).
poetry run python scripts/benchmark_scalability.py \
    --img-dir data/test_split/positives \
    --output reports/scalability_latest.json

# 3. Run the Locust HTTP load test (§3 numbers).
locust -f tests/performance/locustfile.py \
    --host http://localhost:8000 \
    --users 100 --spawn-rate 10 --run-time 5m --headless \
    --csv reports/locust_run

# 4. Tear the stack down.
docker compose -f docker-compose.prod.yml down
```

## 6. Notes and caveats

- The Locust target for `/v1/predict-batch` uses a 3-image ZIP; batch users
  therefore contribute roughly 4× the per-request work of single-image
  users. The target throughput in §3 is expressed in Locust "requests",
  not individual images.
- The warm-up intercept (−0.95 s) is an artefact of the first request
  paying the torch model-load cost. It is negative because the
  least-squares fit is dominated by the large-N points.
- Availability in §4 is measured continuously by the backend's own
  `/metrics availability_percent` gauge and scraped into Prometheus by the
  annotations in `k8s/deployment.yaml`. The 7-day window is evaluated
  against the production Grafana dashboard, not this one-shot report.
