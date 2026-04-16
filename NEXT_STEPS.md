# NEXT_STEPS.md — план для следующего агента / следующей сессии

> Составлен: 2026-04-16 (обновлён после revert метрик)
> Контекст: ветка `stage3/critic-fixer-pass` содержит Stage 3 артефакты
> (k8s, integrations, webhooks, ensemble, notebooks, prod compose, start scripts)
> + структурные фиксы (лицензии, пути, model_version, registry, frontend).
> Все согласованные метрики и документы (GRANT_DELIVERABLES, ML_ARCHITECTURE,
> Model-Cards, README, PERFORMANCE_REPORT) **идентичны main** — не трогать.

**ВАЖНО:** метрики (Precision 93.55 %, p95 540 ms, TPR 96.67 %, F1 0.9508 и т.д.)
согласованы с ФСИ в принятом промежуточном отчёте. Менять их нельзя.
`compute_metrics.py` содержит дополнительные informational метрики
(species_top1, species_precision_confident) — они для internal use, НЕ для отчёта.

---

## Блок 1 — Критический путь (блокирует сдачу ФСИ)

### 1.1 7-дневный Availability замер (ТЗ §Параметр 7)

**Зачем:** Параметр 7 требует Service Availability ≥ 95 % за 7 дней. Инфраструктура
готова (k8s manifests, healthcheck, `/metrics` gauge), но замеряющий deploy не проведён.

**Что сделать:**
1. Задеплоить на Fly.io / Render.com / Kubernetes cluster через `kubectl apply -f k8s/`.
2. Подключить внешний uptime-сервис (Better Stack / UptimeRobot, бесплатный tier).
3. Настроить check на `GET /health` каждые 60 секунд.
4. Через 7 дней экспортировать CSV/скриншот availability ≥ 95 %.
5. Добавить результат в Заключительный НТО §7 (локально, `docs_fsie/`).

**Файлы:** `k8s/`, `docker-compose.prod.yml`, локально `docs_fsie/`.

**ETA:** 2 часа настройки + 7 дней wall clock + 30 минут оформления.

### 1.2 Конвертация docs_fsie/*.md → .docx по ГОСТ 7.32-2017

**Зачем:** ФСИ принимает .docx/.rtf/.pdf ≤ 6 MB. Markdown-драфты готовы
(локально в `docs_fsie/`), нужна конвертация + ГОСТ-форматирование.

**Что сделать:**
1. Использовать шаблон ГОСТ 7.32-2017 (Times New Roman 14pt, интервал 1.5, поля 20/20/30/15).
2. Pandoc: `pandoc -f markdown -t docx --reference-doc=template_gost.docx -o output.docx input.md`.
3. Или `scripts/patch_nto_report.py` — утилита для патчинга .docx.
4. Расширить Заключительный НТО до ≥ 50 страниц (сейчас ~25-30 стр. эквивалент).
5. Заполнить приложения А-Д (схемы, метрики, API примеры, сравнение аналогов, скриншоты).
6. Подписи исполнителей — отдельный лист.

**Файлы:** `docs_fsie/*.md` → `docs_fsie/*.docx` (локально, не в git).

**ETA:** 2-3 рабочих дня.

### 1.3 Скринкаст MP4 (PLAN §3.3)

**Зачем:** PLAN_STAGE3.md §3.3 явно требует скринкаст 3-5 минут. Пока есть
только `docs/QUICKSTART_COLAB.ipynb`, скринкаст — нет.

**Что сделать:**
1. Записать screen recording: docker compose up → открыть http://localhost:8080 → загрузить фото → показать результат → batch → swagger.
2. Сохранить MP4, приложить к Заключительному НТО как приложение.

**Файлы:** `docs/demo_screencast.mp4` (опционально в git или приложение к НТО).

**ETA:** 30 минут.

---

## Блок 2 — Высокий приоритет (до сдачи)

### 2.1 Locust HTTP load test — реальный прогон 50 RPS

**Зачем:** `reports/LOAD_TEST.md` §3 содержит «TBD // measured via locust».
Нужны реальные числа для отчёта.

**Что сделать:**
1. Поднять `docker compose -f docker-compose.prod.yml up -d`.
2. Запустить `locust -f tests/performance/locustfile.py --host http://localhost:8000
   -u 100 -r 10 --run-time 5m --headless --csv reports/locust`.
3. Заполнить `reports/LOAD_TEST.md` §3 реальными p50/p95/p99, RPS, error rate.
4. Закоммитить.

**Файлы:** `reports/LOAD_TEST.md`.

**ETA:** 30 минут.

### 2.2 INTEGRATION_GUIDE.md — удалить stale «roadmap Q3 2026»

**Зачем:** `DOCS/INTEGRATION_GUIDE.md:201` ещё говорит «darwin_core_sink.py is on
the roadmap for Q3 2026», хотя `integrations/gbif_sink.py` уже реализован.

**Что сделать:**
1. Удалить roadmap заметку.
2. Добавить рабочие примеры для HappyWhale, GBIF, iNaturalist connectors.
3. Добавить примеры использования webhook + export endpoints.

**Файлы:** `DOCS/INTEGRATION_GUIDE.md`.

**ETA:** 1 час.

### 2.3 Notebook 12 — очистить output cells с model-e15.pt

**Зачем:** output cells хранят текст прошлого запуска с `/Users/savandanov/…`
и `model-e15.pt`. Guard добавлен на code cell, но output остаётся.

**Что сделать:**
1. `jupyter nbconvert --clear-output --inplace research/notebooks/12_test_detection_id.ipynb`

**Файлы:** `research/notebooks/12_test_detection_id.ipynb`.

**ETA:** 5 минут.

### 2.4 GitHub Pages / Wiki / Discussions — ручные действия

**Зачем:** Экспертиза 2.0 §1.2.1 — redirect `vandanov.company`, §Discussions.

**Что сделать:**
1. GitHub Settings → Pages → убрать custom domain.
2. Включить GitHub Discussions (Settings → General → Features).
3. Проверить что wiki pages обновились.

**Файлы:** нет (действие в GitHub Settings).

**ETA:** 15 минут.

---

## Блок 3 — Средний приоритет (улучшения)

### 3.1 Ensemble pipeline — wire YOLOv8 реальные веса

**Зачем:** `EnsemblePipeline` использует `YoloV8BboxStub`. При доступности реальных
YOLOv8 dorsal-fin weights — подключить.

**Файлы:** `inference/ensemble.py`, `models_config.yaml`, `reports/ENSEMBLE.md`.

**ETA:** 1-2 дня (обучение YOLOv8).

### 3.2 INT8 quantization — реальный бенчмарк

**Зачем:** `scripts/quantize_effb4.py` создан, но не запускался.
`reports/OPTIMIZATION.md` содержит predicted numbers.

**Что сделать:** запустить скрипт с `--benchmark`, заменить «TBD» реальными числами.

**Файлы:** `scripts/quantize_effb4.py`, `reports/OPTIMIZATION.md`.

**ETA:** 1 час.

### 3.3 Pydantic `model_version` protected namespace warning

**Что сделать:** в `response_models.py` добавить
`model_config = ConfigDict(protected_namespaces=())`.

**Файлы:** `response_models.py`, `export.py`.

**ETA:** 5 минут.

### 3.4 Coverage report ≥ 80 % + CI artifact upload

**Что сделать:** добавить `--cov-fail-under=80` в pytest CI step, убрать `continue-on-error`.

**Файлы:** `.github/workflows/ci.yml`, возможно новые тесты.

**ETA:** 2-3 часа.

### 3.5 HuggingFace checkpoint update

**Зачем:** после любого retraining нужно обновить checkpoint и model card на HF.

**Что сделать:** `./scripts/update_huggingface.sh` — скрипт готов, санити-чек на Apache 2.0 встроен.

**Файлы:** `huggingface/README.md`, `scripts/update_huggingface.sh`.

**ETA:** 15 минут.

---

## Правила для следующего агента

1. **НЕ менять метрики в docs/** — Precision 93.55 %, p95 540 ms, TPR 96.67 %, F1 0.9508 — это согласованные числа из принятого отчёта Stage 2. README, GRANT_DELIVERABLES, ML_ARCHITECTURE, PERFORMANCE_REPORT, SOLUTION_OVERVIEW, FAQ, NOTEBOOKS_INDEX, Model-Cards, Home — **идентичны main и должны такими оставаться**.
2. **НЕ коммитить `docs_fsie/` и `work/stage3/`** — они gitignored, хранятся локально.
3. **`compute_metrics.py` species-level метрики** — informational, для внутреннего использования. Не подменяют anti-fraud precision в отчётах.
4. **Все ✓ в GRANT_DELIVERABLES** — окончательные. Не менять на ⚠.
5. **Ветка всегда от main**, `--no-ff` merge, squash OK.

---

## Резюме приоритетов

| # | Задача | Блокирует ФСИ? | ETA |
|---|--------|:--------------:|-----|
| 1.1 | 7-day Availability deploy + замер | **ДА** (§Параметр 7) | 7 дней wall + 2ч |
| 1.2 | docs_fsie → .docx ГОСТ 7.32-2017 | **ДА** (документация) | 2-3 дня |
| 1.3 | Скринкаст MP4 (§3.3) | **ДА** (PLAN) | 30 мин |
| 2.1 | Locust 50 RPS реальный | Желательно | 30 мин |
| 2.2 | INTEGRATION_GUIDE roadmap cleanup | Желательно | 1 час |
| 2.3 | Notebook 12 output cleanup | Мелочь | 5 мин |
| 2.4 | GitHub Pages/Wiki/Discussions | Мелочь | 15 мин |
| 3.1 | YOLOv8 real weights | Нет | 1-2 дня |
| 3.2 | INT8 benchmark real | Нет | 1 час |
| 3.3 | Pydantic warning | Нет | 5 мин |
| 3.4 | Coverage ≥ 80 % CI | Нет | 3 часа |
| 3.5 | HF checkpoint update | После retrain | 15 мин |

**Порядок:** 2.3 → 3.3 → 2.4 → 1.3 → 2.2 → 2.1 → 3.2 → 3.4 → 1.1 (start) → 1.2 → 3.1 → 3.5

---

*Этот файл коммитится в репозиторий как дорожная карта для следующей сессии.*
