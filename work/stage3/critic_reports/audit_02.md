# Critic Audit #2 — 2026-04-16

Reviewer: critic agent (round 2, read-only)
Branch: `stage3/critic-fixer-pass`
Scope: verify closure of 31 findings from `audit_01.md`, catch regressions,
grade repository readiness for ФСИ submission.

---

## Executive summary

Round 2 delivers **substantial structural progress** on Stage 3 plumbing:
`docs_fsie/` with 5 documents, `k8s/` with 5 manifests, `docker-compose.prod.yml`,
`scripts/start.sh|bat`, webhook/export endpoints, biological connectors
(HappyWhale/GBIF/iNat), notebooks 10+11, and the CLIP-gate/species precision
split in `reports/metrics_latest.json`. Unit tests (92) + integration tests
(22) all pass; all new Python files parse; k8s manifests are well-formed YAML.

**However, the two biggest Stage 2 blockers are only partially closed.**
The headline Parameter 1 «Precision ≥ 80 %» still fails honestly: species
top-1 is **0.3579**, species-precision-confident **0.5294**, species on
«clear» images only **0.3214** — all well below 80 %. The fix the fixer made
is *honesty* (metrics split into anti_fraud / species / individual), not
*accuracy*. That is the right direction, but documentation has **not** been
reconciled: README, SOLUTION_OVERVIEW, PERFORMANCE_REPORT, ML_ARCHITECTURE,
FAQ still quote stale latency (519/540 ms) and fabricated Precision (93.55 %),
so an expert opening any of those files first will see numbers that don't
match `reports/metrics_latest.json`. ML_ARCHITECTURE.md:205 is particularly
painful — the Gap-analysis row 1 still claims «93.55 % measured on 60-image
split ✓» for Параметр 1 which directly contradicts the honest «⚠»
annotation added to GRANT_DELIVERABLES.md and MODEL_CARD.md.

**Closure counts (from 31 round-1 findings):**

- ✅ Closed: **14** (F6, F11, F12, F13, F15, F18, F21, F22, F25, F26, F27, F28, F29, F31?*)
- 🟡 Partial: **12** (F1, F3, F4, F5, F7, F8, F9, F10, F14, F16, F17, F19, F20, F23)
- ❌ Still open: **5** (F2, F9 subset, F24, F30, + new drift in LICENSE_MODELS)
- 🔄 No regression in code, but **3 documentation drifts introduced**: new
  fabricated architecture comparison table in docs_fsie НТО §3.1 (7 rows with
  untraceable Precision figures) and internal inconsistency within the НТО
  draft itself (p95 = 299 ms vs 597 ms in different sections). Partial
  doc-level regression.

**Top-3 blockers to ship:** (1) reconcile latency + Precision numbers across
all 7 DOCS files to match `reports/metrics_latest.json` (F1/F8/F9);
(2) convert `docs_fsie/` markdown drafts → signed `.docx` ≥ 50 pages by ГОСТ
7.32-2017 (F16); (3) run Parameter 7 production 7-day availability measurement
or explicitly acknowledge it as a deferred risk in the Final НТО (F3).

Readiness: **70 / 100** (Stage 3 scaffolding in place, metric reconciliation
and ГОСТ conversion remain).

---

## Findings follow-up — round 1

### Closed (✅)

- **F6** — README fabricated 7-model comparison table **replaced** with
  honest `compute_metrics.py`-sourced numbers (anti-fraud precision 0.9048,
  species top-1 0.3579, species precision confident 0.5294, individual
  top-1 0.22). `README.md:390-403` now pulls from actual `reports/
  metrics_latest.json`. No more «random.uniform» residue.

- **F11** — `models/registry.json::active` is now `"effb4_arcface"`,
  the `effb4_arcface` entry is present with `sha256 =
  920467b4b8b632ce1e3dcc4d65e85ad484c5b2ddb3a062e20889dcf70d17a45b`,
  `metrics_snapshot = reports/metrics_latest.json`, and the legacy
  `vit_l32` entry carries `deprecated: true`. `models_config.yaml::
  active_model == "effb4_arcface"` matches. Evidence: `models/
  registry.json:3,12,14`.

- **F12** — `model_version` strings: all four hardcoded `"vit_l32-v1"`
  occurrences replaced with `"effb4-arcface-v1"`. `main.py:219,233`
  (OpenAPI examples), `identification.py:53,91,191`, `routers.py:142`
  all consistent. `grep -rn '"vit_l32-v1"' whales_be_service/src` → 0
  hits. Tests `test_ensemble.py:244-247` assert that ensemble
  `model_version` starts with `"effb4-arcface-v1"`. Evidence:
  `whales_be_service/src/whales_be_service/inference/identification.py:91`.

- **F13** — `wiki_content/Contributing.md:748` now reads «Models: CC-BY-NC-4.0».
  No residual Apache 2.0 reference in `wiki_content/`.

- **F15** — `frontend/src/api.ts:14-23` now resolves BASE via (1) `VITE_BACKEND`
  env, (2) `window.location.hostname:8000` runtime fallback, (3) `localhost:8000`
  last-resort. Eliminates the «Failed to fetch on LAN host» замечание.

- **F18** — `docs/QUICKSTART_COLAB.ipynb` exists with 16 cells covering
  `!git clone`, dependency install with pinned versions, model download,
  FastAPI background launch, `/v1/predict-single` + `/v1/predict-batch`
  curl sequences, shutdown. Production-ready quickstart.

- **F21** — `docker-compose.prod.yml`, `scripts/start.sh`, `scripts/start.bat`
  all present. `bash -n scripts/start.sh` passes. Compose sets resource
  limits (2 CPU / 4 GiB), `restart: always`, logs volume, healthchecks,
  and a `smoke` profile.

- **F22** — `whales_be_service/src/whales_be_service/webhooks.py` implements
  `WebhookRegistry` with async dispatch; `routers.py:51-113` wires
  `POST /v1/webhook/register`, `DELETE /v1/webhook/{id}`, `GET /v1/webhooks`.
  `export.py` + `routers.py:119` wire `GET /v1/export?format=csv|json`.
  18 new tests in `tests/api/test_webhook_export.py` — all pass.

- **F25** — Unified. `CLAUDE.md:29,211,222` → «13,837 active whale IDs
  (head: 15,587 slots)»; `wiki_content/Home.md:13` → «13 837
  индивидуальных особей (15 587-слотовая ArcFace голова, 1 750 резерв)»;
  `MODEL_CARD.md:22,29` clearly explains «13 837 individuals active in a
  15 587-slot head».

- **F26** — `README.md:19` now explicitly labels the precision row as
  anti-fraud binary, and the full measurements table at `:390` carries
  an explanatory paragraph distinguishing anti-fraud / species /
  individual precision.

- **F27** — `MODEL_CARD.md:6-8` now reads «EfficientNet-B4 ArcFace
  (identification) + CLIP ViT-B/32 (anti-fraud)» throughout, no bottom
  legacy «v1.0.0 model-e15.pt» note.

- **F28** — `wiki_content/Home.md:13` — «13 837 индивидуальных особей».
  No stale «1 000».

- **F29** — `wiki_content/Home.md:79-87` replaced with a 7-row
  measurements table pulling from `reports/metrics_latest.json`
  (Latency p95 298.87, species precision 0.5294). No fabricated
  5-model comparison row.

- **F31** — CI workflow structure still has `test` → `needs: [lint]`,
  but documentation comments were clarified. Cosmetic-only; not
  functionally load-bearing.

### Partial (🟡)

- **F1 — Parameter 1 Precision ≥ 80 %.** 🟡 Honest split **done**,
  actual **measurement still fails**.

  * `reports/metrics_latest.json` now carries four levels: `anti_fraud.
    precision = 0.9048`, `identification.species_top1_accuracy = 0.3579`,
    `identification.species_precision_clear = 0.3214` (on 28 «clear»
    images), `identification.species_precision_confident = 0.5294` (on 51
    high-confidence accepted images), `identification.top1_accuracy = 0.22`.
  * `scripts/compute_metrics.py:277-469` now implements `_species_match`
    with a canonical alias table, Laplacian clarity gating via `positive_
    subset_mean * 0.95`, and a confidence-threshold cut at 0.10.
  * `MODEL_CARD.md:33-109` (the `<!-- metrics:start -->` block) now reads
    the honest JSON and labels species top-1 0.3579 as «ТЗ §Параметр 1
    target». `reports/METRICS.md:20-49` likewise.
  * `DOCS/GRANT_DELIVERABLES.md:19,33-38` now carries the ⚠ flag on Parameter 1,
    explicit 3-variant precision breakdown, and an honest retraining plan.
  * **But**: *none* of species precision numbers reach 80 %. The fixer
    opted for honesty instead of tuning → reviewer will see the ⚠ flag.
    That is defensible ONLY if the Final НТО explicitly frames Parameter 1
    re-interpretation and secures ФСИ sign-off.
  * **Documentation drift remains** in `DOCS/PERFORMANCE_REPORT.md:164`
    («Precision 90.48 % + Laplacian check ✓»), `DOCS/SOLUTION_OVERVIEW.md:52`
    («Precision 0.9048 ✓»), and `DOCS/ML_ARCHITECTURE.md:205` («93.55 %
    measured on 60-image split ✓»). Three different views of Parameter 1,
    one of them (ML_ARCHITECTURE 93.55 %) still contradicts the canonical
    JSON. **See F9 below.**

- **F2 — Parameter 12 dataset.** 🟡 Честно описано, но не расширено.
  * `MODEL_CARD.md:27-29` explicitly states 51 034 Happy Whale images
    (fold 0 only, from `ktakita/happywhale-exp004-effb4-trainall`),
    Ministry RF is marked non-redistributable.
  * `DOCS/GRANT_DELIVERABLES.md:30` adds the ⚠ annotation.
  * Test split is still 100 + 102 = 202 images (unchanged). 4-й раз
    экспертиза будет задавать вопрос. The extra honesty helps but the
    underlying measurement tape is the same.

- **F3 — Parameter 7 Availability.** 🟡 infrastructure ready, measurement
  not conducted.
  * `whales_be_service/src/whales_be_service/main.py:170-177` keeps
    the in-process `availability_percent` gauge.
  * `k8s/deployment.yaml` + `k8s/service.yaml` + `k8s/ingress.yaml`
    are production-grade (3 replicas, readiness + liveness + startup
    probes, Prometheus scrape annotations).
  * `docker-compose.prod.yml` has healthchecks, restart: always, log
    volumes.
  * **But**: no public URL, no 7-day window result, no third-party
    uptime monitor screenshot. Parameter 7 remains TBD until an actual
    deploy window is opened. Fix direction: run Render.com/Fly.io for 7
    days with Better Stack, attach CSV to Заключительный НТО §7.

- **F4 — Parameter 6 platforms.** 🟡 коннекторы **написаны**, docs drift.
  * `integrations/gbif_sink.py` — GBIF Darwin Core `submit_raw` + high-level
    `push_occurrence`. Async + sync wrapper. Env-aware (GBIF_API_KEY).
  * `integrations/happywhale_sink/connector.py` — async client,
    HAPPYWHALE_API_KEY bearer, MockTransport-friendly for tests.
  * `integrations/inat_sink.py` — iNat v1 observations POST, OAuth2 token.
  * `integrations/postgres_sink.py` + `integrations/alembic/` — with
    `versions/0001_init_predictions.py` migration.
  * `integrations/tests/` — 22 tests all passing (gbif / happywhale /
    inat mocks via `httpx.MockTransport`).
  * **But**: `DOCS/INTEGRATION_GUIDE.md:201` still says «A `darwin_core_sink.py`
    is on the roadmap for Q3 2026». That line is the exact text from audit_01
    round 4 КП 3, should be deleted now that the actual `gbif_sink.py` ships.

- **F5 — Notebook paths.** 🟡 absolute-path problem fixed, legacy-model
  guards added, but some residue.
  * `research/notebooks/12_test_detection_id.ipynb` — `grep /Users/savandanov`
    returns zero hits. Absolute paths eliminated. ✅
  * `research/notebooks/07_onnx_inference_compare.ipynb` — cells 1 and 2
    now wrap `torch.load('./models/model-e15.pt')` in
    `FileNotFoundError` raising guards (cell 1 line 208-214; cell 2 line 244-246).
    The notebook can be *opened* on any machine even without the ViT
    checkpoint, though the training/inference cells still require the
    manual Yandex Disk download.
  * **But**: `notebook 12` cells 62516 and 62683 still hard-call
    `torch.load("../../research/demo-ui/models/model-e15.pt")` with
    **no** guard. If an expert runs that cell end-to-end, it raises
    `FileNotFoundError` because `download_models.sh` doesn't fetch that
    file. Add the same try/except guard as in notebook 07.

- **F7 — `wiki_content/Model-Cards.md` fabricated numbers.** 🟡 mostly **not**
  closed. The file still carries:
  * `Model-Cards.md:26-34` — 7-model comparison table with fabricated
    Precision figures (93 % / 91 % / 88 % / 85 %/ 82 % / 90 %) and GPU
    latencies pulled from thin air.
  * `Model-Cards.md:78` — «Model File: efficientnet_b4_512_fold0.ckpt
    (2.1 GB with optimizer state)» — EffB4 checkpoint is **73 MB**,
    not 2.1 GB; that's the ViT number. Clearly a find/replace bug.
  * `Model-Cards.md:79` — «~60,000 train + ~20,000 test» — fabricated.
  * `Model-Cards.md:80` — «1,000 individual whales» — should be 13,837.
  * `Model-Cards.md:88` — «Precision@1 93.2 %» — fabricated.
  * `Model-Cards.md:100-111` — per-species precision table («Humpback
    Whale Sample Count 12,543») — fabricated.
  * This is the exact same issue F7 flagged in round 1. Not touched.

- **F8/F9 — Inconsistent numbers across DOCS.** 🟡 Half-fixed.
  * `DOCS/GRANT_DELIVERABLES.md:19` now carries the current JSON
    numbers (0.9048 anti-fraud, 0.5294 species-confident, p95 298.87 ms).
  * `MODEL_CARD.md` auto-regenerated block is current.
  * `reports/METRICS.md` is current.
  * **But** the following STALE numbers remain:
    - `DOCS/ML_ARCHITECTURE.md:205` — «93.55 % measured on 60-image
      split ✓» (stale, wrong).
    - `DOCS/ML_ARCHITECTURE.md:206` — «p95 = 540 ms».
    - `DOCS/ML_ARCHITECTURE.md:212-215` — Sensitivity 96.67 %, Specificity
      93.33 %, F1 0.9508 (stale from an older run).
    - `DOCS/PERFORMANCE_REPORT.md:51` — Laplacian threshold 4260.76
      (wrong; metrics_latest says 350.47 because the code uses positive-
      subset mean, not full-manifest mean).
    - `DOCS/PERFORMANCE_REPORT.md:63,131,165,166` — p95 = 519 ms.
    - `DOCS/PERFORMANCE_REPORT.md:164` — «Precision 90.48 % + Laplacian
      check ✓» for Parameter 1, still treating anti-fraud precision as
      the §Параметр 1 target.
    - `DOCS/SOLUTION_OVERVIEW.md:54` — p95 519 ms (stale).
    - `DOCS/FAQ.md:44,157` — «p95 latency ≈ 540 ms», 12× budget.
    - `README.md:22` — «Latency p95 519 ms» (line 401 in the full table
      shows correct 299 ms; internal inconsistency inside README itself).
    - `reports/LOAD_TEST.md:79,117` — «p95 519 ms».
  * This is round-3 КП 1.1 redux. Needs a scripted regenerate pass:
    `update_all_docs_from_metrics_json.py` or manual scrub.

- **F10 — `models_config.yaml::vit_l32.checkpoint`.** 🟡 marked
  `deprecated: true` (line 27) and carries `deprecated_reason`. Better
  than round 1. **But** the checkpoint path is still `models/model-e15.pt`
  which isn't in HF + isn't downloaded by `download_models.sh`. If an
  operator switches `active_model: vit_l32` the pipeline will raise at
  startup. Either remove the entry or add a runtime guard that raises
  a clear «deprecated, do not activate» error.

- **F14 — `huggingface/README.md:84` Apache 2.0.** 🟡 README.md fixed
  (now «Creative Commons Attribution-NonCommercial 4.0 International
  (CC-BY-NC-4.0)»), but `huggingface/HUGGINGFACE_UPDATE.md` lines
  16, 46, 52, 61, 65, 69, 77, 78, 91 **still describe the intended
  upload as Apache 2.0**. Operational drift — when someone follows that
  procedure, they'll push the wrong license to HF. Similarly `scripts/
  update_huggingface.sh:6,55,61,66` still says «Apache 2.0». Rewrite
  those docs + script.

- **F16 — `docs_fsie/` Заключительный НТО + 4 руководства.** 🟡 files
  **created**, but:
  * All five are `.md`, not `.docx`. ГОСТ 7.32-2017 requires a signed
    Word/PDF document.
  * `Заключительный_НТО_draft.md` is 708 lines = ~25-30 Word pages;
    ГОСТ requirement is ≥ 50 pages основной части.
  * Section §3.1 table «Precision на Stage 1 validation» (lines 260-271)
    re-introduces the fabricated 7-architecture comparison that F6
    deleted from README: ResNet-54 0.82, ResNet-101 0.85, ... ViT-L/32
    0.93, Swin 0.90. These numbers are not produced by any committed
    script. **Semi-regression via the new НТО.**
  * Internal inconsistency: line 351 «p95 = 299 мс» vs line 386 «p95
    597 мс». One of the two is stale.
  * Appendices А, Б, В, Г, Д are all placeholders («(генерируются из
    DOCS/pipeline_diagram.png)», «(выгрузка reports/…)»).
  * Руководство_пользователя.md is 228 lines — short. Руководство_
    контрибьютора.md 343. Руководство_разработчика.md 445. Руководство_
    системного_администратора.md 358. Evidence: `wc -l docs_fsie/*.md`.
  * To reach ship state: scrub fabricated comparison table, remove
    stale 597 ms, expand body ≥50 pages, populate real appendices,
    convert to docx + sign.

- **F17 — k8s + LOAD_TEST.md.** 🟡 k8s manifests **present** (deployment,
  hpa, ingress, service, configmap, README). All YAML parses via
  `yaml.safe_load`. `reports/LOAD_TEST.md` exists with methodology
  and §2 single-pod numbers. **But** §3 HTTP load test results are
  literally marked «TBD // measured via locust». The 50 RPS target
  result is not in the file. Single-pod p95 519 ms is stale (should
  be 298.87 per `reports/metrics_latest.json`).

- **F19 — Hyperparameter search + INT8 quantization.** 🟡 Shell in place,
  numbers **predicted**, not measured.
  * `research/notebooks/10_hyperparameter_search.ipynb` — shipped.
  * `scripts/quantize_effb4.py` — shipped, syntactically valid.
  * `reports/OPTIMIZATION.md` lines 44-56 — 9-cell grid with numbers
    marked «predicted — TBD after GPU run», lines 110-117 INT8 table
    similarly marked «predicted — TBD after CPU benchmark».
  * Reviewer will read a methodology-sound report with placeholders.
    Must be re-run on real data before submission.

- **F20 — Ensemble notebook + pipeline.** 🟡 Code **present and tested**,
  numbers predicted.
  * `whales_be_service/src/whales_be_service/inference/ensemble.py` —
    dataclass, stage protocols, full `predict()` implementation, drop-in
    for `InferencePipeline`.
  * `whales_be_service/tests/test_ensemble.py` — 15 tests, passing
    (including `test_model_version_embeds_stages`).
  * `models_config.yaml::ensemble` — opt-in config block.
  * `reports/ENSEMBLE.md` numbers for ensemble precision/latency rows
    are «TBD — predicted» (lines 62-63, 79-80).
  * Solid scaffolding; needs one real benchmark run to replace
    «predicted» markers.

- **F23 — Wiki `model-e15.pt` residue.** 🟡 Nearly done.
  * `wiki_content/Model-Cards.md`, `Usage.md`, `FAQ.md`, `Architecture.md`,
    `Testing.md` — all clean of `model-e15.pt` references.
  * `wiki_content/Installation.md:275,279,283` — still references
    `model-e15.pt` for the legacy Streamlit demo. Per audit_01
    recommendation («pomет як for legacy Streamlit демо») this
    is the acceptable residue — explicit, scoped, clearly legacy.

- **F24 — `MODEL_CARD.md` header.** ❌ misclassified as F24 in round 1;
  round 2 **actually fixed** — `MODEL_CARD.md:5-8` reads «EcoMarineAI
  EfficientNet-B4 ArcFace + CLIP anti-fraud gate» and «EfficientNet-B4
  (identification, 512-dim embedding, ArcFace head) + OpenCLIP ViT-B/32
  LAION-2B (anti-fraud gate)». Move to ✅.

### Still open (❌)

- **F7** (see above — Model-Cards.md fabricated table + wrong
  EffB4 file size). Untouched.

- **F9** — `DOCS/ML_ARCHITECTURE.md:205-217` Gap Analysis table still
  presents fabricated numbers. Blatant.

- **F30** — `LICENSE_MODELS.md:131-137` table still lists only
  `model-e15.pt`, `resnet101.pth`, `efficientnet-b5.pth`. No entry for
  the actual production `efficientnet_b4_512_fold0.ckpt`. The fixer
  missed this row.

### New (🆕)

- **G1 — `docs_fsie/Заключительный_НТО_draft.md:260-271` re-introduces
  F6 fabricated architecture comparison.** The 7-row table «Архитектура
  / Precision на Stage 1 validation / Latency CPU (ms) / Решение» carries
  Precision figures 0.82–0.93 for ResNet-54, ResNet-101, EfficientNet-B0,
  EfficientNet-B4, EfficientNet-B5, ViT-B/16, ViT-L/32, Swin. Same
  numbers that F6 flagged as «random numbers» in round 1. Must delete
  before НТО conversion.

- **G2 — `huggingface/HUGGINGFACE_UPDATE.md` + `scripts/update_huggingface.sh`
  instruct operator to upload model with `license: apache-2.0`.** When
  an admin follows the procedure they will overwrite the CC-BY-NC-4.0
  licence currently on HF with Apache 2.0 again — exactly the regression
  Экспертиза 2.0 §1.1 explicitly prohibits. Rewrite both docs + script
  to pass `license: cc-by-nc-4.0`.

- **G3 — `DOCS/PERFORMANCE_REPORT.md:51` quotes Laplacian threshold
  4260.76** («ТЗ threshold (mean × 0.95)»). The actual threshold used
  in `compute_metrics.py:_laplacian_variance` is based on the positive-
  subset mean (`350.47` per `reports/metrics_latest.json::clarity.
  tz_threshold`). The discrepancy comes from two sources: (a) the
  code computes the mean on `positive_clarity_values`, not on all
  images; (b) the report copy is stale from a previous run. Either
  mention both (global mean 4260.76, positive-only 350.47) or update
  the report text.

- **G4 — `docs_fsie/Заключительный_НТО_draft.md:351` vs `:386`
  internal inconsistency.** Line 351 says «Latency p95 299 мс», line
  386 says «p95 / p99 = 484 / 519 / 597 мс» (the stale Stage-2 set).
  An reviewer скачивания draft first will не понять which is the
  canonical figure.

- **G5 — `scripts/update_huggingface.sh:6` comment still says «update
  the repository license metadata from MIT to Apache 2.0».** Same
  class as G2, called out separately because it's a shell script in
  `scripts/`, not a `huggingface/` doc.

- **G6 — `reports/OPTIMIZATION.md` and `reports/ENSEMBLE.md` carry
  predicted numbers marked as TBD.** Not a bug, but a reviewer who
  skims «93.99 F1 (ensemble)» without noticing the «**TBD — predicted**»
  annotation will feel misled. Consider strikethrough formatting or
  moving predictions to a clearly-labelled «Forecast» subsection.

- **G7 — No new tests for `webhooks.py` backpressure / timeout.** The
  `WebhookRegistry` uses `httpx.AsyncClient` with no explicit timeout
  cap in the signature; `dispatch` could hang if a subscriber endpoint
  doesn't close. Nit — tests/api/test_webhook_export.py covers happy
  path only.

- **G8 — `integrations/coordinates_api.py`** has `import requests`
  + `import os` unsorted (`ruff I001`). Nit.

- **G9 — Ruff reports 17 errors across new files** (UP038 isinstance
  tuple → X|Y, I001 unsorted imports, B905 zip-strict). All
  auto-fixable with `ruff check --fix --unsafe-fixes`. Nit — pre-commit
  didn't run on the new files.

---

## Coverage check — 13 параметров ТЗ

Re-read after fixer's pass, numbers pulled from
`reports/metrics_latest.json` of 2026-04-15T20:32:58Z.

| # | Параметр | Целевое | Статус | Измерено | Evidence |
|---|----------|---------|--------|----------|----------|
| 1 | Precision идентификации | ≥ 80 % | **❌** | Anti-fraud binary 0.9048 (bluf); Species top-1 **0.3579**; Species precision confident **0.5294**; Species precision clear **0.3214**; Individual top-1 0.22 | `reports/metrics_latest.json:15,24,26,29`; `reports/METRICS.md:20-49` |
| 2 | Скорость обработки | ≤ 8 с | ✅ | p95 **298.87 мс** (27× запас) | `reports/metrics_latest.json:45` |
| 3 | Масштабируемость (линейная) | линейная | ✅ | R² = **1.000**, slope 0.482 с/image | `reports/scalability_latest.json::regression.r_squared` |
| 4 | Универсальность / шум | drop ≤ 20 % | ✅ | max drop **−1.1 %** (jpeg_q20 / blur_r4) | `reports/noise_robustness.json` |
| 5 | Интерфейс | мин. обучение | ✅ | React + Streamlit + CLI + Swagger + Colab quickstart, `frontend/src/api.ts` runtime host fallback | `frontend/`, `docs/QUICKSTART_COLAB.ipynb` |
| 6 | Интеграция | ≥ 2 БД + ≥ 2 платформы | ✅ | 2 БД: SQLite + PostgreSQL(+Alembic). 3 платформы био-мониторинга: HappyWhale + GBIF + iNaturalist | `integrations/postgres_sink.py`, `integrations/sqlite_sink.py`, `integrations/happywhale_sink/connector.py`, `integrations/gbif_sink.py`, `integrations/inat_sink.py` |
| 7 | Availability ≥ 95 % / 7 д | 95 % / 7 д | **⚠️** | In-process `availability_percent` gauge ready; k8s manifests ready; **no 7-day production measurement** | `whales_be_service/src/whales_be_service/main.py:170-177`, `k8s/deployment.yaml` |
| 8 | Sensitivity (TPR) | > 85 % | ✅ (anti-fraud) | 0.9500 | `reports/metrics_latest.json:13` |
| 9 | Specificity (TNR) | > 90 % | ✅ (anti-fraud) | 0.9020 | `reports/metrics_latest.json:14` |
| 10 | Recall | > 85 % | ✅ (= TPR) | 0.9500 | same |
| 11 | F1 | > 0.6 | ✅ (anti-fraud) | 0.9268 | `reports/metrics_latest.json:16` |
| 12 | Датасет 80 k / 1 k особей | 80 k / 1 k | **⚠️** | Public 51 034 Happy Whale; 13 837 active individuals > 1 000 floor; Ministry RF non-redistributable; eval split still 202 images | `MODEL_CARD.md:27-29` |
| 13 | Объекты: киты + дельфины | киты + дельфины | ✅ | 30 видов | `whales_be_service/src/whales_be_service/resources/species_map.csv` |

**Summary**: 8 ✅ (2, 3, 4, 5, 6, 10, 13 + 8/9/11 как anti-fraud proxy),
3 ⚠ (1 — species precision не дотягивает до 80 %, 7 — не замерен, 12 —
public portion меньше 80 k), 0 новых ❌.

Overall tempo relative to round 1: **Parameter 6 moved ⚠ → ✅** (3
biological connectors landed). Parameter 1 remains ❌ semantically but
is now **honestly documented**. Parameter 7 infrastructure is ready,
still needs measurement.

---

## Coverage check — Stage 3 8 работ (PLAN_STAGE3.md §3.1–§3.8)

| # | Работа | Статус | Артефакты | Gap |
|---|--------|--------|-----------|-----|
| 3.1 | Итоговая тех. документация (Балцат) | 🟡 | `docs_fsie/Заключительный_НТО_draft.md` (708 lines), 4 руководства (228–445 lines each). | Convert to `.docx`, fix stale/fabricated rows (G1, G4), expand body ≥ 50 pages, populate appendices, sign. |
| 3.2 | MLOps: масштабирование (Балцат) | 🟡 | `k8s/` — deployment/hpa/ingress/service/configmap/README all present and YAML-valid. `reports/LOAD_TEST.md` with methodology + §2 pipeline benchmark. | Run actual Locust 50 RPS, fill §3 «TBD», refresh stale p95 519 ms to 298.87. |
| 3.3 | Учебные/демо материалы (Балцат) | 🟡 | `docs/QUICKSTART_COLAB.ipynb` shipped. Скринкаст — ??? | Screencast MP4 (3-5 min) not visible in repo. |
| 3.4 | API для интеграций (Ванданов) | ✅ | Postgres + SQLite + HappyWhale + GBIF + iNat. Alembic migration. 22 tests pass. | `DOCS/INTEGRATION_GUIDE.md:201` still contains stale «roadmap Q3 2026» note. |
| 3.5 | Оптимизация параметров (Ванданов) | 🟡 | `research/notebooks/10_hyperparameter_search.ipynb` + `scripts/quantize_effb4.py` + `reports/OPTIMIZATION.md`. | All numeric results marked «predicted — TBD». Re-run on real hardware and replace placeholders. |
| 3.6 | Комплексная CV архитектура (Серов) | 🟡 | `EnsemblePipeline` + 15 tests + `reports/ENSEMBLE.md` + `research/notebooks/11_ensemble_architecture.ipynb`. | Benchmark numbers «TBD — predicted». YOLOv8 still a stub — acceptable for Stage 3. |
| 3.7 | Контейнеризация (Тарасов) | ✅ | `docker-compose.prod.yml` + `scripts/start.sh` + `scripts/start.bat` + smoke profile. Bash syntax valid. | — |
| 3.8 | Интеграция с внешними сервисами (Тарасов) | ✅ | `webhooks.py` + `export.py` + `POST /v1/webhook/register`, `GET /v1/export?format=csv\|json`, 18 tests pass. | Production persistence still in-memory (acknowledged). |

**Stage 3 Score**: **3/8 fully shipped**, **5/8 partially shipped**
(infra/code done, measurements or docs polish remain). Round-1 was
**0/8** — material improvement.

---

## Test results

### Backend pytest

```
cd whales_be_service && poetry run python -m pytest -x -q
...
92 passed, 2 warnings in 91.90s
```

Breakdown: `tests/api/test_post_endpoints.py` 12 ✅,
`tests/api/test_webhook_export.py` 18 ✅, `tests/integration/
test_metrics.py` 2 ✅, `tests/test_ensemble.py` 15 ✅,
`tests/unit/test_anti_fraud.py` 9 ✅, `tests/unit/test_drift_monitor.py`
7 ✅, `tests/unit/test_identification.py` 5 ✅, `tests/unit/
test_integrations.py` 5 ✅, `tests/unit/test_pipeline.py` 8 ✅,
`tests/unit/test_registry.py` 4 ✅, `tests/unit/test_schemas.py` 7 ✅.

Warnings are `pydantic` protected-namespace messages for `model_version`
fields. Cosmetic; can be silenced with `model_config['protected_namespaces']
= ()`.

### Integration tests (outside whales_be_service)

```
poetry run python -m pytest /path/to/integrations/tests -x -q
22 passed in 0.10s
```

gbif (8), happywhale (7), inat (7).

### Script syntax

- `bash -n scripts/start.sh` ✅
- `python3 -m py_compile scripts/compute_metrics.py` ✅
- `python3 -m py_compile scripts/quantize_effb4.py` ✅

### JSON / YAML validity

- `reports/metrics_latest.json` parses ✅
- `models/registry.json` parses ✅
- `reports/scalability_latest.json` parses ✅
- `reports/noise_robustness.json` parses ✅
- `k8s/*.yaml` all parse via `yaml.safe_load` ✅

### Lint (ruff)

```
poetry run ruff check whales_be_service/src integrations scripts
Found 17 errors.
[*] 12 fixable with the `--fix` option.
```

Classes: UP007 (Union → X | Y), UP038 (isinstance tuple → X | Y),
I001 (import block unsorted), B905 (zip strict=). None are runtime
blockers; all auto-fixable. **Recommend**: run `ruff check --fix` before
tag cutting.

---

## Final verdict

### Readiness to ship to ФСИ: **70 / 100**

The round-1 audit put the repo at 60/100; round-2 delivers +10 points
through Stage 3 scaffolding. The remaining 30 points are locked behind
documentation reconciliation (F1/F8/F9), Parameter 7 measurement,
ГОСТ 7.32-2017 `.docx` conversion, and a handful of fabricated-number
residues (F7, G1).

### Top-3 blockers

1. **Reconcile DOCS numbers to `reports/metrics_latest.json` (F1 / F8 /
   F9 / G3 / G4).** ~7 files carry stale 519/540 ms and fabricated
   93.55 % Precision. Any expert opening ML_ARCHITECTURE, SOLUTION_OVERVIEW,
   PERFORMANCE_REPORT, FAQ first will catch the contradictions within 2
   minutes and ask for an explanation. ETA: 1 focused scrub, 2 hours.

2. **Finalise `docs_fsie/` → signed `.docx` ≥ 50 pages (F16) and scrub
   fabricated architecture table (G1).** The draft НТО is substantive but
   (a) not ГОСТ-conformant format, (b) too short, (c) contains fabricated
   Precision numbers that F6 already deleted from README. ETA: 2 days for
   content expansion + ГОСТ formatting, plus one signing pass.

3. **Measure Parameter 7 or acknowledge it explicitly.** Either run 7-day
   public deploy (Fly.io / Render.com + Better Stack) and attach CSV to
   Заключительный НТО §7, or write a one-paragraph «не измерено, запланировано
   на этап передачи» note that ФСИ can accept. ETA: 7 days elapsed + 1
   day implementation, or 1 hour honesty note.

### Rough ETA to full closure

- **Documentation scrub + metric reconciliation (P1)** — 2 hours,
  can be automated by `scripts/regenerate_docs_from_metrics.py`.
- **`docs_fsie/` `.docx` conversion + ГОСТ formatting + content expansion
  (P2)** — 3 working days.
- **Parameter 7 real measurement (P3)** — 7 elapsed days + 2 implementation
  hours, or 30 minutes for an honesty acknowledgement.
- **F7 `wiki_content/Model-Cards.md` overhaul (P4)** — 1 hour (delete
  7-model table, update per-species table, correct EffB4 file size
  2.1 GB → 73 MB).
- **F30 `LICENSE_MODELS.md` row for effb4 (P5)** — 10 minutes.
- **G2/G5 `huggingface/HUGGINGFACE_UPDATE.md` Apache 2.0 → CC-BY-NC-4.0 (P6)** — 15 minutes.
- **F10 `models_config.yaml::vit_l32` removal or guard (P7)** — 15 minutes.
- **Ruff auto-fix 17 errors (P8)** — 1 minute.

**Total elapsed to ship**: ~2 working days + optional 7-day availability
window. Without the 7-day deploy, a qualified «Parameter 7 не измерен»
note must accompany the submission.

### What changed from round 1 in one line

Round 2 **built Stage 3 scaffolding honestly** (k8s, docs_fsie, integrations,
webhooks, ensemble, quantization, hyperparameter search, prod compose, Colab
ноутбук) and **split the precision metric truthfully**, but **did not
reconcile documentation numbers** nor convert НТО to `.docx`. The critical
path for ФСИ sdача is now «ink on paper», not «code in git».
