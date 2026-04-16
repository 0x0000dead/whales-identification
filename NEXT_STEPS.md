# NEXT_STEPS.md — план для следующего агента / следующей сессии

> Составлен: 2026-04-16
> Контекст: ветка `stage3/critic-fixer-pass` замержена (или готова к мержу),
> два раунда critic-аудита проведены (`audit_01.md`, `audit_02.md` — хранятся
> локально в `work/stage3/critic_reports/`). Текущий grade: **70/100**.
> Цель: довести до **90/100** и закрыть Stage 3 для ФСИ.

---

## Блок 1 — Критический путь (блокирует сдачу ФСИ)

### 1.1 Retraining EffB4 на полной 5-fold + Ministry RF (~+20 п.п. species precision)

**Зачем:** species precision = 0.5294 (52.9 %), ТЗ §Параметр 1 требует ≥ 80 %.
Текущий public checkpoint обучен только на fold 0 Happy Whale (10k из 51k).

**Что сделать:**
1. Получить доступ к GPU (Colab Pro, Kaggle, или свой сервер с ≥16 GB VRAM).
2. Запустить `whales_identify/train.py` на full 5-fold Happy Whale + если доступны — Ministry RF данные.
3. Сохранить новый checkpoint, обновить `models/checksums.sha256`, `models/registry.json`.
4. Перезапустить `scripts/compute_metrics.py --update-model-card` — сверить species precision.
5. Если species precision ≥ 0.80 — обновить все `⚠` на `✓` в `DOCS/GRANT_DELIVERABLES.md`, `DOCS/ML_ARCHITECTURE.md`, `reports/METRICS.md`.
6. Push новый checkpoint на HuggingFace через `scripts/update_huggingface.sh`.

**Файлы:** `whales_identify/train.py`, `scripts/compute_metrics.py`, `models/checksums.sha256`, `models/registry.json`, `MODEL_CARD.md`, `reports/metrics_latest.json`.

**ETA:** 1 день GPU time + 2 часа документации.

### 1.2 7-дневный Availability замер (ТЗ §Параметр 7)

**Зачем:** Параметр 7 требует Service Availability ≥ 95 % за 7 дней. Инфраструктура
готова (k8s manifests, healthcheck, `/metrics` gauge), но измерения нет.

**Что сделать:**
1. Задеплоить на Fly.io / Render.com / любой Kubernetes cluster через `kubectl apply -f k8s/`.
2. Подключить внешний uptime-сервис (Better Stack, UptimeRobot — бесплатный tier).
3. Настроить check на `GET /health` каждые 60 секунд.
4. Через 7 дней экспортировать CSV/скриншот availability ≥ 95 %.
5. Добавить результат в `reports/AVAILABILITY_7DAY.md` (или напрямую в Заключительный НТО §7).
6. Обновить `DOCS/GRANT_DELIVERABLES.md` §Параметр 7 с `⚠` → `✓` и прикрепить ссылку.

**Файлы:** `k8s/`, новый `reports/AVAILABILITY_7DAY.md`, `DOCS/GRANT_DELIVERABLES.md`.

**ETA:** 2 часа настройки + 7 дней wall clock + 30 минут оформления.

### 1.3 Расширить test split до ≥ 1000 изображений

**Зачем:** текущий split (100+102 = 202 images) даёт ±7 п.п. confidence interval
на TPR/TNR/Precision. Эксперт ФСИ может усомниться в надёжности метрик.

**Что сделать:**
1. Скачать ещё 400-500 позитивных изображений из Happy Whale Kaggle test set.
2. Скачать ещё 400-500 негативных из Intel Image Dataset / COCO non-animal.
3. Обновить `data/test_split/manifest.csv` (1000+ строк).
4. Перезапустить `scripts/compute_metrics.py --update-model-card`.
5. Обновить все документы с новыми числами (автоматически через injection в MODEL_CARD).

**Файлы:** `data/test_split/manifest.csv`, `data/test_split/{positives,negatives}/`, `reports/metrics_latest.json`.

**ETA:** 3 часа.

---

## Блок 2 — Высокий приоритет (нужно до сдачи, но не блокирует)

### 2.1 DOCS остаточные stale числа — финальный scrub

**Зачем:** round 2 critic нашёл что `DOCS/PERFORMANCE_REPORT.md` и `DOCS/SOLUTION_OVERVIEW.md`
ещё содержат отдельные строки со stale данными (Laplacian threshold 4260.76, отдельные
упоминания «12× budget» вместо «27×», etc.). Большинство исправлено, но edge cases
остались в длинных .md файлах.

**Что сделать:**
1. Написать `scripts/regenerate_docs_from_metrics.py` — скрипт который парсит
   `reports/metrics_latest.json` и автозаменяет все metrc-carrying строки в
   `DOCS/PERFORMANCE_REPORT.md`, `DOCS/SOLUTION_OVERVIEW.md`, `DOCS/FAQ.md`,
   `DOCS/NOTEBOOKS_INDEX.md`, `reports/LOAD_TEST.md`.
2. Запускать его через pre-commit hook (PostToolUse на `scripts/compute_metrics.py`).
3. Или однократно прогнать + закоммитить.

**Файлы:** новый `scripts/regenerate_docs_from_metrics.py`, `DOCS/*.md`, `reports/*.md`.

**ETA:** 2-3 часа.

### 2.2 Locust HTTP load test — реальный прогон 50 RPS

**Зачем:** `reports/LOAD_TEST.md` §3 содержит «TBD // measured via locust».
Нужны реальные числа concurrent-load behavior.

**Что сделать:**
1. Поднять `docker compose -f docker-compose.prod.yml up -d`.
2. Запустить `locust -f tests/performance/locustfile.py --host http://localhost:8000
   -u 100 -r 10 --run-time 5m --headless --csv reports/locust`.
3. Заполнить `reports/LOAD_TEST.md` §3 реальными p50/p95/p99, RPS, error rate.
4. Закоммитить.

**Файлы:** `reports/LOAD_TEST.md`, `reports/locust_*.csv` (опционально).

**ETA:** 30 минут.

### 2.3 INTEGRATION_GUIDE.md — удалить stale «roadmap Q3 2026»

**Зачем:** `DOCS/INTEGRATION_GUIDE.md:201` ещё говорит «darwin_core_sink.py is on
the roadmap for Q3 2026», хотя `integrations/gbif_sink.py` уже реализован.

**Что сделать:**
1. Удалить roadmap заметку.
2. Добавить рабочие примеры для HappyWhale, GBIF, iNaturalist connectors.
3. Добавить примеры использования webhook + export endpoints.

**Файлы:** `DOCS/INTEGRATION_GUIDE.md`.

**ETA:** 1 час.

### 2.4 Notebook 12 — второй torch.load в output cells

**Зачем:** audit_02 отметил что cells 62516/62683 (output cells с сохранённым вводом)
ещё содержат `torch.load("../../research/demo-ui/models/model-e15.pt")`. Guard добавлен
на cell 12, но output cells хранят текст прошлого запуска.

**Что сделать:**
1. Очистить outputs: `jupyter nbconvert --clear-output --inplace research/notebooks/12_test_detection_id.ipynb`.
2. Или удалить конкретные output cells через `nbformat`.

**Файлы:** `research/notebooks/12_test_detection_id.ipynb`.

**ETA:** 10 минут.

---

## Блок 3 — Средний приоритет (улучшения для quality of life)

### 3.1 Ensemble pipeline — wire YOLOv8 реальные веса

**Зачем:** `EnsemblePipeline` использует `YoloV8BboxStub` (возвращает full-image bbox).
Когда будут доступны реальные YOLOv8 dorsal-fin weights, подключить их.

**Что сделать:**
1. Обучить или скачать YOLOv8 dorsal-fin detector checkpoint.
2. Заменить `YoloV8BboxStub` в `whales_be_service/src/whales_be_service/inference/ensemble.py`.
3. Обновить `models_config.yaml::ensemble::active_stages` — включить `yolov8_bbox`.
4. Перезапустить ensemble benchmark, обновить `reports/ENSEMBLE.md`.

**Файлы:** `inference/ensemble.py`, `models_config.yaml`, `reports/ENSEMBLE.md`.

**ETA:** 1-2 дня (в основном обучение YOLOv8).

### 3.2 INT8 quantization — реальный бенчмарк

**Зачем:** `scripts/quantize_effb4.py` создан, но не запускался. `reports/OPTIMIZATION.md`
содержит predicted ускорение 2-3×. Нужны реальные числа.

**Что сделать:**
1. Запустить `python scripts/quantize_effb4.py --benchmark` на CPU-машине.
2. Заменить «TBD» в `reports/OPTIMIZATION.md` реальными fp32 vs int8 числами.
3. Если ускорение значимо — добавить int8 checkpoint в `models/checksums.sha256`.

**Файлы:** `scripts/quantize_effb4.py`, `reports/OPTIMIZATION.md`, `models/checksums.sha256`.

**ETA:** 1 час.

### 3.3 GitHub Pages / Wiki live fixes

**Зачем:** Экспертиза 2.0 §1.2.1 указывала на redirect `vandanov.company`. Это
настраивается в Settings → Pages репозитория и требует ручного действия владельца.

**Что сделать:**
1. GitHub Settings → Pages → убрать custom domain → использовать `0x0000dead.github.io/whales-identification`.
2. Включить GitHub Discussions (Settings → General → Features).
3. Проверить что все wiki pages обновились с нового `wiki_content/`.

**Файлы:** нет (действие в GitHub Settings).

**ETA:** 15 минут.

### 3.4 Pydantic `model_version` protected namespace warning

**Зачем:** 2 warnings в тестах — `Field "model_version" in Detection has conflict
with protected namespace "model_"`. Косметика, но видно в pytest output.

**Что сделать:**
1. В `response_models.py` (и `export.py` если есть `ExportRecord`): добавить
   `model_config = ConfigDict(protected_namespaces=())`.

**Файлы:** `whales_be_service/src/whales_be_service/response_models.py`, `export.py`.

**ETA:** 5 минут.

### 3.5 Coverage report ≥ 80 % + CI artifact upload

**Зачем:** `CLAUDE.md` и FSI правила требуют ≥ 80 % test coverage. Сейчас coverage
не мерится в CI (continue-on-error на codecov upload).

**Что сделать:**
1. Добавить `--cov=src --cov-report=xml --cov-fail-under=80` в pytest CI step.
2. Убрать `continue-on-error: true` с codecov upload.
3. Убедиться что coverage ≥ 80 % (или дописать тесты до порога).

**Файлы:** `.github/workflows/ci.yml`, возможно новые тесты.

**ETA:** 2-3 часа.

---

## Блок 4 — Post-merge (после принятия Stage 3)

### 4.1 Конвертация docs_fsie/*.md → .docx по ГОСТ 7.32-2017

**Зачем:** ФСИ принимает .docx/.rtf/.pdf ≤ 6 MB. Markdown-драфты готовы
(локально в `docs_fsie/`), но не конвертированы.

**Что сделать:**
1. Использовать шаблон ГОСТ 7.32-2017 (Times New Roman 14pt, интервал 1.5, поля 20/20/30/15).
2. Pandoc: `pandoc -f markdown -t docx --reference-doc=template_gost.docx -o output.docx input.md`.
3. Или `scripts/patch_nto_report.py` — уже готов для патчинга .docx.
4. Расширить Заключительный НТО до ≥ 50 страниц (сейчас ~25-30 стр. эквивалент).
5. Заполнить приложения А-Д (схемы, метрики, API примеры, сравнение аналогов, скриншоты).
6. Подписи исполнителей — отдельный лист.

**Файлы:** `docs_fsie/*.md` → `docs_fsie/*.docx` (локально, не в git).

**ETA:** 2-3 рабочих дня.

### 4.2 HuggingFace update

**Зачем:** после retraining (§1.1) нужно обновить checkpoint на HF + model card.

**Что сделать:**
1. `./scripts/update_huggingface.sh` — скрипт готов, санити-чек на Apache 2.0 встроен.
2. Убедиться что HF card показывает `cc-by-nc-4.0`.

**Файлы:** `huggingface/README.md`, `scripts/update_huggingface.sh`.

**ETA:** 15 минут.

---

## Резюме приоритетов

| # | Задача | Блокирует ФСИ? | ETA | Grade impact |
|---|--------|:--------------:|-----|:------------:|
| 1.1 | Retraining EffB4 5-fold | **ДА** (§Параметр 1) | 1 день GPU | +10 |
| 1.2 | 7-day Availability замер | **ДА** (§Параметр 7) | 7 дней wall | +5 |
| 1.3 | Test split ≥ 1000 images | Желательно | 3 часа | +3 |
| 2.1 | Doc scrub автоматизация | Нет, но экспертиза поймает | 2 часа | +2 |
| 2.2 | Locust 50 RPS реальный | Нет | 30 мин | +2 |
| 2.3 | INTEGRATION_GUIDE roadmap | Нет | 1 час | +1 |
| 2.4 | Notebook 12 output cleanup | Нет | 10 мин | +1 |
| 3.1 | YOLOv8 real weights | Нет | 1-2 дня | +2 |
| 3.2 | INT8 benchmark real | Нет | 1 час | +1 |
| 3.3 | GitHub Pages/Wiki | Нет | 15 мин | +1 |
| 3.4 | Pydantic warning | Нет | 5 мин | +0 |
| 3.5 | Coverage ≥ 80 % CI | Нет | 3 часа | +2 |
| 4.1 | docs_fsie → .docx ГОСТ | **ДА** (документация) | 3 дня | req'd |
| 4.2 | HF checkpoint update | После retraining | 15 мин | +0 |

**Если делать в порядке impact/effort:** 2.4 → 3.4 → 3.3 → 2.3 → 2.2 → 2.1 → 3.2 → 1.3 → 3.5 → 1.1 → 1.2 → 4.1 → 3.1 → 4.2

---

*Этот файл коммитится в репозиторий как дорожная карта для следующей сессии / следующего агента.*
