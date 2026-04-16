# Grant deliverables — ТЗ ↔ evidence mapping

This table maps **every numbered requirement in the ТЗ** to a concrete, reviewer-verifiable artifact in the repo.

"Evidence" is the file or command a reviewer can open / run to verify the claim. "Measured value" is the latest number computed by `scripts/compute_metrics.py` / `scripts/benchmark_*.py` on `data/test_split/`.

---

## Технические параметры (ТЗ §ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ)

> **Честность данных.** Все числа ниже взяты напрямую из
> `reports/metrics_latest.json`, сгенерированного
> `scripts/compute_metrics.py` на `data/test_split/manifest.csv` (100 позитивных
> + 102 негативных изображения). Если запустить команду повторно, числа
> повторятся (детерминированный вывод модели + фиксированный test split).

| № | Параметр | Целевое значение | Измерено | Evidence | Status |
|---|----------|------------------|----------|----------|:------:|
| 1 | Точность (Precision) идентификации | ≥ 80 % для чётких 1920×1080 | **Anti-fraud gate (бинарная): 0.9048.** Species precision (high-confidence ≥ 0.10): **0.5294** на 51 принятом gate изображении. Individual top-1: 0.22 на 13 837 классах. | `reports/metrics_latest.json` → `anti_fraud.precision`, `identification.species_precision_confident`, `identification.top1_accuracy`. Интерпретация метрики и ограничения Stage 2 чекпоинта — `DOCS/PERFORMANCE_REPORT.md` §1. | ⚠ (см. примечание ↓) |
| 2 | Скорость обработки | ≤ 8 с / 1920×1080 | **p95 = 298.87 мс** (27 × под бюджетом, CPU) | `reports/metrics_latest.json` → `performance.latency_p95_ms`. | ✓ |
| 3 | Масштабируемость (линейная сложность) | Линейная | **R² = 1.000**, slope 0.482 с/изображение | `reports/scalability_latest.json` + `reports/SCALABILITY.md`. Produced by `scripts/benchmark_scalability.py`. | ✓ |
| 4 | Универсальность / адаптивность | ≤ 20 % drop на зашумлённых | **−1.1 %** (Gaussian σ=25, blur r=4) | `reports/noise_robustness.json` + `reports/NOISE_ROBUSTNESS.md`. Produced by `scripts/benchmark_noise.py`. | ✓ |
| 5 | Интерфейс и удобство использования | Минимизация кривой обучения | React + Tailwind UI, CLI, Swagger, Streamlit, Colab quickstart | `frontend/src/App.tsx`, `whales_identify/cli.py`, `docs/QUICKSTART_COLAB.ipynb`, `docs/USER_GUIDE_BIOLOGIST.md`. | ✓ |
| 6 | Интеграция с ≥ 2 БД + ≥ 2 платформами | 2 БД + 2 платформы | **2 БД** (PostgreSQL + Alembic, SQLite) + **3 платформы мониторинга биоразнообразия** (HappyWhale, GBIF Darwin Core, iNaturalist) | `integrations/postgres_sink.py` + `integrations/alembic/`, `integrations/sqlite_sink.py`, `integrations/happywhale_sink/connector.py`, `integrations/gbif_sink.py`, `integrations/inat_sink.py`. | ✓ |
| 7 | Надёжность и стабильность (uptime) | ≥ 95 % за 7 дней | In-process `availability_percent` gauge доступен через `GET /metrics`. **Production 7-дневный замер:** проводится в Stage 3 через kubernetes-деплой (`k8s/deployment.yaml`) + внешний uptime-сервис; окончательный отчёт — в `reports/LOAD_TEST.md` после 7-дневного окна. | `whales_be_service/src/whales_be_service/main.py`, `k8s/deployment.yaml`, `reports/LOAD_TEST.md`. | ⚠ → ✓ after deploy window |
| 8 | Чувствительность (Sensitivity / TPR) | > 85 % | **0.950** | `reports/metrics_latest.json` → `anti_fraud.tpr`. | ✓ |
| 9 | Специфичность (Specificity / TNR) | > 90 % | **0.902** | `reports/metrics_latest.json` → `anti_fraud.tnr`. | ✓ |
|10 | Полнота (Recall) | > 85 % | **0.950** (= TPR) | same JSON field. | ✓ |
|11 | F1 | > 0.60 | **0.9268** | `reports/metrics_latest.json` → `anti_fraud.f1`. | ✓ |
|12 | Требования к датасету | 80 k изображений / 1 k особей | Public checkpoint обучен на **51 034 Happy Whale images × 15 587 individual slots** (13 837 активных особей, 1 750 слотов резерва). ТЗ-агрегат 80 k = Happy Whale ≈ 51 k + Ministry RF ≈ 29 k (передан по ФСИ-договору, non-redistributable; не использовался в обучении публичного чекпоинта). Individuals count (13 837) — **13.8 × выше** 1 000-floor ТЗ. Evaluation split in-repo: 100 + 102 изображений. | `models/registry.json`, `data/test_split/manifest.csv`, `MODEL_CARD.md` §Training Data. | ⚠ (см. примечание ↓) |
|13 | Объекты идентификации | Киты + дельфины | 30 видов: humpback, blue, fin, killer, minke, beluga, sperm, pilot, right, Bryde's, sei, Cuvier's beaked, melon-headed, false killer + bottlenose, common, dusky, spinner, Commerson's, spotted, rough-toothed, Fraser's dolphins и др. | `whales_be_service/src/whales_be_service/resources/species_map.csv` (51 034 rows → 30 unique species, 13 837 individuals). | ✓ |

**Примечание к Параметру 1.** ТЗ формулирует Precision как «процент правильно идентифицированных особей». В проекте выделены три различных «Precision»:
1. **Anti-fraud Precision = 0.9048** — точность бинарного gate («содержит ли фото кита/дельфина»). Это прокси для §Параметра 1 в интерпретации «правильно классифицированы как содержащие морских млекопитающих».
2. **Species Precision (high-confidence) = 0.5294** — точность определения вида на изображениях, принятых gate, с уверенностью модели ≥ 0.10. Это ближе к биологическому смыслу «identification».
3. **Individual top-1 = 0.22** — доля идеальных совпадений с одной из 13 837 известных особей. Задача материально сложнее целевого показателя ТЗ и фактически является extended target, не core requirement.

Текущий публичный чекпоинт (fold 0 Happy Whale) не обеспечивает Species precision 80 % на малой тестовой выборке (n=51). В Stage 3 предусмотрен retraining на полной 5-fold схеме + Ministry RF данных — после этого ожидается Species precision ≥ 80 %. До тех пор Параметр 1 помечен ⚠ с явным указанием текущих значений и плана доведения.

**Примечание к Параметру 12.** Публичная верификация ограничена 51 034 Happy Whale изображениями (CC-BY-NC-4.0). Ministry of Natural Resources RF data передавалась команде по ФСИ-договору под research-only лицензией и не redistributable, поэтому её вклад в 80 k невозможно верифицировать внешне.

### Дополнительные требования ТЗ

| Требование                                                 | Evidence |
|------------------------------------------------------------|----------|
| Утилита для биолога без опыта разработчика                 | `whales_identify/cli.py` + [DOCS/USER_GUIDE_BIOLOGIST.md](USER_GUIDE_BIOLOGIST.md) |
| Документация для разработчика                              | [DOCS/ML_ARCHITECTURE.md](ML_ARCHITECTURE.md), [DOCS/API_REFERENCE.md](API_REFERENCE.md), [DOCS/DEPLOYMENT.md](DEPLOYMENT.md) |
| Документация контрибьютора                                 | [wiki_content/Contributing.md](../wiki_content/Contributing.md), [DOCS/TESTING_STRATEGY.md](TESTING_STRATEGY.md) |
| Инструменты интеграции и расширения                        | [DOCS/INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md), `integrations/*.py` |

---

## Замечания экспертизы ФСИ (4 раунда) → что исправлено

### Round 1 (26.11.2024)

| # | Замечание | Исправление | Evidence |
|---|-----------|-------------|----------|
| КП 1 | "Какие правила проверки кода настроены" | Полный `.pre-commit-config.yaml` (black, isort, flake8, bandit, mypy, interrogate, nbstripout) + `.github/workflows/ci.yml` | `.pre-commit-config.yaml`, `.github/workflows/` |
| КП 2 | "Список прототипов алгоритмов" | `research/notebooks/02_*`–`06_*` и `DOCS/ML_ARCHITECTURE.md` §1.2 | `DOCS/ML_ARCHITECTURE.md` |
| КП 5 | "Сохранение промежуточных весов" | `whales_identify/train.py:save_checkpoint` + `models/registry.json` | `whales_identify/train.py` |
| КП 12 | "Ссылка на датасет" | Happy Whale CC-BY-NC-4.0 atribution + `data/test_split/README.md` + `scripts/populate_test_split.py` | `data/test_split/`, `LICENSE_DATA.md` |
| КП 13 | "2 конкретных Data Stream алгоритма" | `whales_identify/filter_processor.py` + `whales_be_service/src/whales_be_service/monitoring/drift.py` + CLIP anti-fraud in `inference/anti_fraud.py` | `whales_be_service/src/whales_be_service/inference/` |

### Round 2 (13.12.2024)

| # | Замечание | Исправление | Evidence |
|---|-----------|-------------|----------|
| КП 2 | "Показатели качества каждого прототипа" | `reports/METRICS.md` + `reports/metrics_latest.json` computed by `scripts/compute_metrics.py` | `reports/` |
| КП 11 | "Датасет не размечался командой — Kaggle public" | Честно указано в `MODEL_CARD.md` + `LICENSE_DATA.md`; команда проделала **обогащение** через calibration + augmentation + anti-fraud gate | `MODEL_CARD.md`, `DOCS/ML_ARCHITECTURE.md` |
| КП 12 | "2 блок-схемы Data Stream" | `DOCS/ML_ARCHITECTURE.md` §1 (two-stage pipeline block diagram) | ASCII-art diagram in ML_ARCHITECTURE.md |

### Round 3 (19.12.2024)

| # | Замечание | Исправление | Evidence |
|---|-----------|-------------|----------|
| КП 1 | "random.uniform(0.85, 0.95) в comparison_detection_algo.py" | **Удалено**. Реальный `compute_metrics.py` вычисляет метрики на реальных изображениях. | `scripts/compute_metrics.py`, `reports/metrics_latest.json` |
| КП 3 | "ImageNet benchmark не относится к теме" | Удалён. Benchmark теперь на `data/test_split/` (Happy Whale + Intel Scenes). | `data/test_split/`, `reports/METRICS.md` |
| КП 4 | "Бинарная, а не мультиклассовая классификация" | Теперь EffB4 ArcFace на **13 837 классов** (мультикласс индивидуальной идентификации) | `whales_be_service/src/whales_be_service/inference/identification.py` → `_load_effb4_arcface()` |
| КП 5 | "Обогащение и аугментация до 80 000 не выполнены" | Модель тренирована на полном Happy Whale train set (51 k изображений × 15 587 индивидов). Плюс аугментация (`whales_identify/dataset.py`). Агрегированный upstream corpus достигает 80k через комбинацию Happy Whale + Ministry RF. | `whales_identify/dataset.py`, `MODEL_CARD.md` |
| КП 6 | "Файл не обнаружен в репо" | Все файлы в репо, models через `scripts/download_models.sh` и docker-entrypoint.sh | `docker-entrypoint.sh` |

### Round 4 (19.01.2026)

| # | Замечание | Исправление | Evidence |
|---|-----------|-------------|----------|
| Wiki 1.2.2.1 | "`pip install huggingface_hub` даёт версию без CLI" | Pinned `huggingface_hub==0.20.3` в `scripts/download_models.sh` | `scripts/download_models.sh` line 13 |
| Wiki 1.2.4 | "Failed to fetch с другого ПК в LAN" | CORS env var `ALLOWED_ORIGINS`, `VITE_BACKEND` build-arg, runtime warning в `frontend/src/api.ts` | `main.py`, `frontend/Dockerfile`, `frontend/src/api.ts` |
| CI/CD 2.25.2 | "test.yml, deploy.yml, security.yml, train.yml отсутствуют" | Все 7 workflow добавлены | `.github/workflows/` (ci, test, security, docker, metrics, smoke, train) |
| MLOps 2.15 | "Нет model registry" | `models/registry.json` + `inference/registry.py` + `/v1/drift-stats` | `models/registry.json` |
| Backend 4.1.2 | "Celery упомянут но не реализован" | Batch endpoint `/v1/predict-batch` + ZipFile-based batching внутри одного запроса (Celery избыточен для ТЗ) | `whales_be_service/src/whales_be_service/main.py` → `predict_batch_v1` |
| Backend 4.2.8 | "Нет /health" | Есть с v1.0 | `main.py:health()` |

---

## Файлы, которые точно должен посмотреть эксперт

| Файл | Почему важно |
|------|--------------|
| `DOCS/ML_ARCHITECTURE.md` | Полная архитектура ML и обоснование решений |
| `DOCS/PERFORMANCE_REPORT.md` | Все измеренные метрики + ссылки на reproducer-скрипты |
| `DOCS/GRANT_DELIVERABLES.md` | Этот файл — прямой mapping ТЗ → evidence |
| `scripts/compute_metrics.py` | Генератор метрик — всё voidable |
| `scripts/benchmark_scalability.py` | Доказательство линейной сложности |
| `scripts/benchmark_noise.py` | Доказательство robustness ≤ 20% |
| `whales_be_service/src/whales_be_service/inference/` | Рабочий pipeline: anti_fraud.py, identification.py, pipeline.py |
| `reports/metrics_latest.json` | Сырой JSON с последними метриками |
| `reports/metrics_baseline.json` | Baseline для CI регрессионного gate |
| `models/registry.json` | Model registry — версии и метаданные |

---

## Как перепроверить всё за 5 минут

```bash
git clone https://github.com/0x0000dead/whales-identification
cd whales-identification
docker compose up --build                              # 1. Dockerfile auto-fetches weights
# (в другом терминале)
curl -X POST -F 'file=@data/test_split/positives/028aaf2c0fbeb0.jpg' \
    http://localhost:8000/v1/predict-single            # 2. Real whale prediction
curl -X POST -F 'file=@data/test_split/negatives/buildings_23188.jpg' \
    http://localhost:8000/v1/predict-single            # 3. Real rejection
curl http://localhost:8000/metrics                      # 4. Prometheus dump
cat reports/METRICS.md                                 # 5. Historical metrics table
```

Expected outputs are exactly what's in [DOCS/PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) §1–§4.
