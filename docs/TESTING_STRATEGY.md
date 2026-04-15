# Testing strategy

The repo has **three layers** of automated tests, run at different frequencies:

1. **Unit tests** — fast (< 5 s), run on every commit.
2. **API integration tests** — medium (< 10 s), use a `StubPipeline` so they don't need real ML weights.
3. **ML integration tests** — slow (30–60 s), require real weights and the full test split; run only on `main`.

Plus one manual layer:

4. **End-to-end smoke test** (`scripts/smoke_test.sh`) — runs curl against a real container, used in CI on push to main.

Total count: **88 passing tests** as of the latest run (2 deselected slow).

---

## Layer 1: Unit tests

Location: `whales_be_service/tests/unit/` and `whales_identify/tests/`.

Scope: every component of the inference pipeline in isolation, no network, no weights, no torch where avoidable.

| File                              | Tests | Focus                                                  |
|-----------------------------------|------:|--------------------------------------------------------|
| `test_schemas.py`                 |     7 | Dataclass immutability, enum values                    |
| `test_drift_monitor.py`           |     7 | Rolling window, alarm heuristic, singleton            |
| `test_anti_fraud.py`              |     8 | Prompt inventory, degraded-mode, threshold loader     |
| `test_pipeline.py`                |     8 | All 8 branching cases in InferencePipeline            |
| `test_identification.py`          |     5 | Backend routing priority (effb4 > vit > resnet > err) |
| `test_registry.py`                |     4 | Model registry JSON parsing                            |
| `whales_identify/tests/test_model.py` |    14 | GeM, ArcMarginProduct, CetaceanIdentificationModel |
| `whales_identify/tests/test_dataset.py` |    ~10 | Dataset + augmentation pipeline                    |
| `whales_identify/tests/test_config.py`  |    ~5  | Training config loader                             |
| `whales_identify/tests/test_utils.py`   |    ~5  | Helpers                                            |
| `whales_identify/tests/test_cli.py`     |     4  | CLI subcommands with stub pipeline                |

### Design principles

- **No real ML weights in unit tests.** Heavy loads are covered by layer 3.
- **Mock at module boundaries.** `InferencePipeline` tests mock `AntiFraudGate` and `IdentificationModel`; `AntiFraudGate` tests mock `open_clip`.
- **Degraded-mode tests are mandatory.** Every lazy-loaded component must have a test proving that it fails safe (logs + returns permissive result) when its heavy dep is absent.
- **Frozen dataclasses.** `GateResult` and `PredictionResult` are tested to raise on assignment so downstream code can assume immutability.

### Running

```bash
pytest -m "not slow"           # from repo root — uses pytest.ini
pytest -m "not slow" -q        # quiet
pytest whales_be_service/tests/unit/test_pipeline.py  # just one file
```

---

## Layer 2: API integration tests

Location: `whales_be_service/tests/api/test_post_endpoints.py`.

Scope: end-to-end through FastAPI `TestClient`, but with a `StubPipeline` injected via `app.dependency_overrides` (see `whales_be_service/tests/conftest.py`).

The stub returns:
- red images → anti-fraud rejected
- any other image → accepted, fixed humpback_whale ID

Why a stub? Because loading real CLIP + EffB4 inside pytest makes the suite take > 30 s. The API contract (status codes, headers, schema) is what we test here, not ML accuracy.

| Test                                         | What it proves                                         |
|----------------------------------------------|--------------------------------------------------------|
| `test_health_check`                          | `/health` returns 200 `{"status":"ok"}`                |
| `test_predict_single_accepted`               | Accepted schema has all 11 Detection fields            |
| `test_predict_single_rejected_returns_200`   | Anti-fraud rejection returns HTTP 200, not 4xx         |
| `test_predict_single_unsupported_media`      | 415 on non-image content type                          |
| `test_predict_single_empty_file`             | 400 on zero-byte upload                                |
| `test_predict_batch_success`                 | ZIP with 3 images → 3 Detection objects (mixed a/r)    |
| `test_predict_batch_wrong_content_type`      | 415                                                    |
| `test_predict_batch_bad_zip`                 | 400                                                    |
| `test_predict_batch_empty_zip`               | 200 with empty list                                    |
| `test_v1_predict_single` / `test_v1_predict_batch` | v1 alias routes also work                       |
| `test_metrics_endpoint`                      | Prometheus text contains all expected counters         |

---

## Layer 3: ML integration tests

Location: `whales_be_service/tests/integration/test_metrics.py`.

Scope: pull the real pipeline, run `scripts/compute_metrics.py` on a small sample (50 images), assert TZ targets:

- TNR ≥ 0.90
- TPR ≥ 0.85
- top-1 accuracy ≥ 0.0 (placeholder; raised to 0.5 once the full-fold model is wired in)

Marked `@pytest.mark.slow` + `@pytest.mark.integration` — excluded by the default `pytest -m "not slow"` filter. CI runs them only on `push to main`:

```bash
pytest -m slow
```

If the test split isn't populated, the tests `pytest.skip()` rather than fail — so forks without Happy Whale credentials still get a green CI.

---

## Layer 4: End-to-end smoke test

`scripts/smoke_test.sh` runs against a live server (uvicorn or docker-compose) and validates:

1. `GET /health` → 200
2. `POST /v1/predict-single` with a red-noise image → 200 with full Detection schema
3. `POST /v1/predict-batch` with a 2-image ZIP → 200 with 2-element list
4. `GET /metrics` → text with all expected counters
5. `GET /v1/drift-stats` → JSON

Wired into CI via `.github/workflows/smoke.yml` on every push to main. The workflow starts docker-compose, waits for the backend health check, runs the script, prints logs on failure, and tears down.

---

## Measured performance of the test suite

| Suite                  | Tests | Duration (CPU) |
|------------------------|------:|---------------:|
| Unit + API + CLI       |    88 |         ~4.4 s |
| Integration (slow)     |     2 |        ~45.0 s |
| Smoke test             |     — |        ~15.0 s |

---

## Coverage snapshot

Run locally:

```bash
cd whales_be_service && poetry run pytest --cov=src --cov-report=term
```

Target: ≥ 70 % line coverage on `whales_be_service/src/whales_be_service/inference/*`. Enforced via `.pre-commit-config.yaml` → `interrogate` (docstring coverage) + `pytest-cov` in CI.

---

## CI matrix

| Workflow               | Trigger                 | Coverage      |
|------------------------|-------------------------|---------------|
| `.github/workflows/ci.yml`       | push + PR (main/develop) | lint, unit, API, docker build |
| `.github/workflows/test.yml`     | push + PR               | unit tests (belt-and-suspenders over ci.yml) |
| `.github/workflows/metrics.yml`  | push main               | compute_metrics + regression gate |
| `.github/workflows/smoke.yml`    | push main               | docker compose + smoke_test.sh |
| `.github/workflows/security.yml` | push + PR               | bandit + pip-audit |
| `.github/workflows/train.yml`    | manual                  | `whales_identify.train` launcher |
| `.gitlab-ci.yml`                 | push (mirror)           | parallel mirror of ci.yml       |

---

## How to add a new test

1. Unit test lives next to its target:  `whales_be_service/tests/unit/test_<module>.py`.
2. Name the function `test_<behaviour_under_conditions>`.
3. Mock any heavy dependency (`torch`, `open_clip`, `rembg`) via `monkeypatch.setitem(sys.modules, ..., ...)` or `MagicMock`.
4. Add one "happy path" test + one "error path" test per function.
5. Run `pytest -m "not slow" -q` to verify.
6. Push to a branch, open a PR — CI runs the full matrix.
