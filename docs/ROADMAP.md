# Roadmap

This roadmap maps the state of the repo onto the three **этапа** of the ФСИ grant and the technical parameters of the ТЗ.

## Этап 1 — Исследование и прототипирование (done)

| КП работа                                                   | Status  | Evidence |
|-------------------------------------------------------------|:-------:|----------|
| 1.1 Настройка репозитория + авто-проверки PR                | ✓       | `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `pytest.ini` |
| 1.2 Тестирование пилотных прототипов детекции               | ✓       | `research/notebooks/06_benchmark_*`, `comparison_research/` |
| 1.3 Интервью с экспертами                                   | ✓       | `wiki_content/Contributing.md`, списки экспертов |
| 1.4 Разработка прототипов детекции + обработки              | ✓       | `whales_identify/filter_processor.py`, `dataset.py`, `research/notebooks/01_*` |
| 1.5 Обучение НС с сохранением промежуточных весов           | ✓       | `whales_identify/train.py` + `huggingface/` checkpoints |
| 1.6 Исследование алгоритмов CV                              | ✓       | `research/notebooks/02_*`–`05_*` (ViT, EfficientNet, Swin, ResNet) |
| 1.7 Прототипы алгоритмов ML для идентификации               | ✓       | `whales_identify/model.py` (CetaceanIdentificationModel) |
| 1.8 Общая архитектура ML-системы                            | ✓       | [DOCS/ML_ARCHITECTURE.md](ML_ARCHITECTURE.md) |
| 1.9 Системный анализ + оптимизация                          | ✓       | `whales_be_service/src/whales_be_service/inference/` refactor |
| 1.10 Многоклассовая классификация                           | ✓       | EfficientNet-B4 ArcFace 13 837 классов |
| 1.11 Код-ревью прототипов                                   | ✓       | `.github/workflows/ci.yml` + pre-commit |
| 1.12 Тестирование и сравнение архитектур                    | ✓       | `reports/METRICS.md`, `DOCS/PERFORMANCE_REPORT.md` |
| 1.13 Сбор, обогащение и аугментация данных                  | ✓       | `whales_identify/dataset.py`, Happy Whale train set |
| 1.14 Data Stream алгоритмы ML                               | ✓       | `whales_identify/filter_processor.py`, `monitoring/drift.py` |

## Этап 2 — Интеграция и валидация (done)

| КП работа                                               | Status  | Evidence |
|---------------------------------------------------------|:-------:|----------|
| 2.1 Открытый репозиторий + согласование лицензий        | ✓       | `LICENSE`, `LICENSE_DATA.md`, `LICENSE_MODELS.md`, `LICENSES_ANALYSIS.md` |
| 2.2 CI/CD + MLOps                                       | ✓       | 7 GH workflows (ci / metrics / smoke / security / docker / train / lint), `models/registry.json`, `monitoring/drift.py` |
| 2.3 Тестирование с обратной связью                      | ✓       | `whales_be_service/tests/` (88 unit tests), `whales_identify/tests/` |
| 2.4 Backend + ML-модели                                 | ✓       | FastAPI + `InferencePipeline` (real inference, no mocks) |
| 2.5 Валидация ML-алгоритмов, выбор финального решения   | ✓       | `scripts/compute_metrics.py` + `reports/metrics_baseline.json` |
| 2.6 Пользовательский интерфейс                          | ✓       | `frontend/` (React + Tailwind + RejectionCard + ConfidenceGauge) |
| 2.7 Интеграция BE + FE + browser/mobile                 | ✓       | docker-compose, `VITE_BACKEND`, nginx.conf, responsive Tailwind |

## Этап 3 — Развёртывание и масштабирование (in progress)

| КП работа                                               | Status  | Next step |
|---------------------------------------------------------|:-------:|-----------|
| 3.1 Итоговая техническая документация                   | ✓       | 15 docs under `DOCS/` — this file, ML_ARCHITECTURE, PERFORMANCE_REPORT, API_REFERENCE, etc. |
| 3.2 MLOps для высокой нагрузки                          | ✓       | `monitoring/drift.py`, `/v1/drift-stats`, availability gauge, baseline regression gate |
| 3.3 Учебные и демо материалы                            | ✓       | `research/demo-ui/`, `README.md` «для бабушки», `DOCS/USER_GUIDE_BIOLOGIST.md` |
| 3.4 API для взаимодействия FE с ML                      | ✓       | `/v1/predict-single`, `/v1/predict-batch`, `/v1/drift-stats` |
| 3.5 Эксперименты с оптимизацией параметров              | ✓       | `scripts/calibrate_clip_threshold.py`, `scripts/benchmark_*.py` |
| 3.6 Комплексная архитектура CV                          | ✓       | CLIP gate + EffB4 ArcFace + confidence gating |
| 3.7 Контейнеризация + file запуска                      | ✓       | `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh` (auto-download from HF) |
| 3.8 Интеграция с внешними сервисами                     | ✓       | `integrations/sqlite_sink.py`, `integrations/postgres_sink.py`, CSV export via CLI, HF Hub mirror |
| 3.X Мобильная версия UI (добавлено по замечанию ФСИ)    | partial | Tailwind responsive — done; PWA wrapper planned Q3 |

## Beyond the grant — потенциал глобального помощника

| Milestone                                         | ETA     | Notes |
|---------------------------------------------------|---------|-------|
| GPU inference mode                                | Q2 2026 | ONNX export + TensorRT — already demonstrated in `research/notebooks/07_onnx_inference_compare.ipynb` |
| Bbox detector (YOLOv8-cetacean)                   | Q2 2026 | Train on `data/backfin_annotations.csv` (5 201 rows) + re-label |
| Top-k results via nearest-neighbour retrieval     | Q2 2026 | Pre-compute embeddings for all training images, use FAISS |
| Real-time video stream processing                 | Q3 2026 | Frame sampling + batched inference |
| Mobile-native app (Flutter)                       | Q3 2026 | REST API is already mobile-friendly |
| Federated learning for private fleet datasets     | Q4 2026 | See Yandex ML blog ref in the review notes |
| Biologist feedback loop (active learning)         | Q4 2026 | Collect disagreements, re-train head monthly |
| Public leaderboard + community dashboard          | 2027    | Every new contributor gets a DOI |
| OpenAPI SDKs (Python, R, Julia)                   | 2027    | Auto-generated from `/openapi.json` |
| UN SDG 14 integration (GBIF + OBIS sync)          | 2027    | One-way upload of anonymised observations |

## Community extensibility

The project is designed so that a third party can:

1. **Add a new species.** Drop images into a training directory, update `species_map.csv`, re-fit the ArcFace head (no backbone retraining needed).
2. **Add a new integration sink.** Copy `integrations/sqlite_sink.py`, swap the driver — takes under an hour.
3. **Deploy to another country's cloud.** `docker-entrypoint.sh` downloads weights from any HF repo via the `HF_REPO` env var. Fork the HF repo, retrain, point `HF_REPO=yourorg/their-whales` → done.
4. **Run entirely offline.** Mount a pre-populated `models/` directory in the Docker container; the entrypoint skips the download.

See [CONTRIBUTING.md](../wiki_content/Contributing.md) and [FAQ.md](FAQ.md) for contributor workflow.
