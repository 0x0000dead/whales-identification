# EcoMarineAI — План работ: закрытие 2 этапа + выполнение 3 этапа

> Составлен: 2026-04-15  
> Автор: Серов А.И.  
> Статус: к исполнению

---

## ЧАСТЬ 1 — Незакрытые задачи 2 этапа

### 1.1 CI/CD — smoke test (🔴 критично, исправляется сейчас)

**Проблема:** `smoke.yml` падает с `ModuleNotFoundError: No module named 'PIL'`.  
Скрипт `scripts/smoke_test.sh` генерирует тестовые изображения через PIL, но в GitHub Actions runner Pillow не установлен.

**Исправление:** добавить шаг `pip install Pillow` в `.github/workflows/smoke.yml` перед запуском smoke теста.  
**Файл:** `.github/workflows/smoke.yml`  
**Статус:** исправлено в текущей сессии, ожидает commit + push.

---

### 1.2 SHA256 проверка весов модели (🟠 высокий)

**Проблема:** `scripts/download_models.sh` скачивает модели без проверки целостности. При повреждённой загрузке CI проходит, но модель даёт неверные результаты.

**Что нужно:**
- Добавить SHA256-проверку после скачивания каждого файла
- Добавить retry (3 попытки с backoff) при ошибке
- Создать `models/checksums.sha256` с эталонными хешами

**Файлы:** `scripts/download_models.sh`, `models/checksums.sha256`

---

### 1.3 Ноутбуки — устаревшие пути (🟠 высокий)

**Проблема 1:** `research/notebooks/07_onnx_inference_compare.ipynb` — заголовок ссылается на `model-e15.pt` (устаревшая модель, не скачивается автоматически).  
**Исправление:** обновить описание ноутбука под актуальную модель `efficientnet_b4_512_fold0.ckpt`.

**Проблема 2:** `research/notebooks/12_test_detection_id.ipynb` — захардкожен абсолютный путь  
`/Users/savandanov/Documents/Github/whales-identification/...`  
Ноутбук не запускается ни у кого кроме С.А. Ванданова.  
**Исправление:** заменить на относительный путь `../../models/efficientnet_b4_512_fold0.ckpt`.

**Файлы:** `research/notebooks/07_onnx_inference_compare.ipynb`, `research/notebooks/12_test_detection_id.ipynb`

---

### 1.4 models_config.yaml — несоответствие checkpoint (🟡 средний)

**Проблема:** `models_config.yaml` для модели `vit_l32` указывает `checkpoint: "models/model-e15.pt"` — файл, которого нет в HuggingFace и который не скачивается автоматически.  
**Исправление:** изменить на `checkpoint: "models/efficientnet_b4_512_fold0.ckpt"` либо создать отдельную запись для legacy ViT-модели со статусом `deprecated`.

**Файл:** `models_config.yaml`

---

### 1.5 GitHub — ручные действия (🟢 низкий)

Требуют прав владельца репозитория (`0x0000dead`):

| Действие | Где | Зачем |
|----------|-----|-------|
| Включить Discussions | Settings → Features → Discussions ✓ | Ссылка в документации |
| Убрать redirect на vandanov.company | Settings → Pages → Custom domain: очистить | GitHub Pages работает |

---

## ЧАСТЬ 2 — Работы 3 этапа (по ТЗ)

### 3.1 Итоговая техническая документация (Балцат К.И.)

По условиям договора с ФСИ требуется предоставить:

#### Заключительный НТО

Структура строго по `инструкция_заключительная.doc` и ГОСТ 7.32-2017:

```
- Список исполнителей (с подписями)
- Реферат (заполняется в системе ФСИ отдельно)
- Содержание
- Нормативные ссылки (ГОСТ 34.601-90, ГОСТ 7.32-2017, ГОСТ Р ИСО/МЭК 25010-2015)
- Определения (датасет, детекция, идентификация, метрическое обучение, ...)
- Обозначения и сокращения (ИИ, CV, API, CI/CD, TPR, TNR, F1, ...)
- Введение
- Раздел 1: Сбор и обработка данных (КП 3.1 Тарасов А.А.)
- Раздел 2: Алгоритмы CV и обработка изображений (КП 3.2 Балцат К.И.)
- Раздел 3: ML-модели и оптимизация (КП 3.5 Ванданов С.А.)
- Раздел 4: Масштабируемая архитектура (КП 3.6 Серов А.И.)
- Раздел 5: Пользовательский интерфейс и документация (КП 3.3 Балцат К.И.)
- Раздел 6: API и интеграции (КП 3.4, 3.8 Ванданов С.А., Тарасов А.А.)
- Раздел 7: Тестирование, контейнеризация, DevOps (КП 3.7 Тарасов А.А.)
- Заключение (выводы + оценка соответствия всем 11 параметрам ТЗ)
- Список источников (по ГОСТ Р 7.0.5-2008)
- Приложения (скриншоты, метрики, схемы)
```

Требования:
- Минимум **50 страниц** основной части (без рисунков и таблиц)
- Формат: `.doc`, `.rtf` или `.pdf`; размер ≤ **6 МБ**
- PDF только «обычный» (не сканы), вставка PDF-изображений не допускается

#### Четыре руководства (отдельные документы)

| Документ | Аудитория | Ключевое содержание |
|----------|-----------|---------------------|
| Руководство пользователя | Биологи без IT-опыта | Установка, загрузка фото, интерпретация результатов, FAQ |
| Руководство системного администратора | DevOps/IT | Docker deploy, переменные среды, мониторинг `/metrics`, backup |
| Руководство разработчика | Python-разработчики | Архитектура, добавление новой модели, обучение на своих данных, API |
| Руководство контрибьютора | Open-source контрибьюторы | Git workflow, pre-commit, тесты, PR процесс, Code of Conduct |

Создать в `docs_fsie/` как `.docx` файлы.

---

### 3.2 MLOps: масштабирование под высокую нагрузку (Балцат К.И.)

**Цель по ТЗ:** линейная временная сложность, Availability ≥ 95% за 7 дней.

**Что сделать:**
- `k8s/deployment.yaml` — Kubernetes Deployment с `replicas: 3`
- `k8s/hpa.yaml` — HorizontalPodAutoscaler (CPU > 70% → scale up до 10 реплик)
- `k8s/ingress.yaml` — Nginx Ingress с rate limiting (60 req/min per IP)
- Load test: Locust, достичь **50 RPS** при p95 latency < 8s
- Результаты нагрузочного теста в `reports/LOAD_TEST.md`

---

### 3.3 Учебные и демо-материалы (Балцат К.И.)

**Что сделать:**
- `docs/QUICKSTART_COLAB.ipynb` — Google Colab ноутбук «Быстрый старт»: подключиться к API, загрузить фото, получить результат (5 минут для нового пользователя)
- Скринкаст (MP4, 3–5 минут) — демонстрация полного workflow через веб-интерфейс
- Обновить `research/demo-ui/streamlit_app.py` — добавить пояснения для биологов, примеры видов

---

### 3.4 API для интеграций (Ванданов С.А.)

**Цель по ТЗ:** совместимость с ≥ 2 базами данных и ≥ 2 платформами мониторинга.

**Что сделать:**
- `integrations/happywhale_sink/connector.py` — рабочий коннектор к HappyWhale API (авторизация, отправка наблюдения)
- `integrations/gbif_sink.py` — новый: отправка записей в GBIF (Global Biodiversity Information Facility)
- `integrations/postgres_sink.py` — дооформить: Alembic-миграции, документация
- `integrations/inat_sink.py` — новый: iNaturalist API коннектор
- Добавить в README раздел «Интеграции» с примерами

Платформы: **HappyWhale** + **GBIF** (≥ 2 ✓)  
Базы данных: **PostgreSQL** + **SQLite** (≥ 2 ✓)

---

### 3.5 Оптимизация параметров моделей (Ванданов С.А.)

**Что сделать:**
- `research/notebooks/10_hyperparameter_search.ipynb` — grid search по `min_confidence` (0.05 / 0.10 / 0.15) и CLIP threshold (0.45 / 0.55 / 0.65) с таблицей Precision/Recall/F1
- INT8 quantization EfficientNet-B4 через `torch.quantization` — измерить ускорение vs точность
- Результаты зафиксировать в `reports/OPTIMIZATION.md`

---

### 3.6 Комплексная CV-архитектура (Серов А.И.)

**Что сделать:**
- `research/notebooks/11_ensemble_architecture.ipynb` — сравнение:  
  Single model (EfficientNet-B4) vs Ensemble (CLIP gate + EfficientNet-B4 + YOLOv8 bbox)
- Реализовать в `whales_be_service/src/.../inference/identification.py` поддержку режима `ensemble` через `models_config.yaml`:
  ```yaml
  active_model: ensemble
  models:
    ensemble:
      mode: ensemble
      stages: [clip_gate, effb4_arcface, yolov8_bbox]
  ```
- Benchmark: точность и latency ensemble vs single, зафиксировать в `reports/ENSEMBLE.md`

---

### 3.7 Контейнеризация и файл запуска (Тарасов А.А.)

**Что сделать:**
- `docker-compose.prod.yml` — production compose: resource limits, restart: always, volume для логов
- `scripts/start.sh` — единый скрипт запуска (Linux/Mac):
  1. Проверить Docker
  2. Скачать модели (`./scripts/download_models.sh`)
  3. Поднять сервисы (`docker compose up -d`)
  4. Проверить health (`curl /health`)
- `scripts/start.bat` — аналог для Windows (WSL2 / Docker Desktop)
- Smoke test через docker-compose profile: `docker compose --profile smoke run test`

---

### 3.8 Интеграция с внешними сервисами (Тарасов А.А.)

**Что сделать:**
- Webhook endpoint `POST /v1/webhook/register` — регистрация callback URL для push-уведомлений о результатах
- Export endpoint `GET /v1/export?format=csv|json` — выгрузка истории предсказаний
- Документация `docs/INTEGRATION_GUIDE.md` — обновить с примерами для каждой интеграции

---

## ЧАСТЬ 3 — Что нужно для успешного закрытия проекта

### Чеклист сдачи Stage 3 в ФСИ

```
[ ] Заключительный НТО.docx (≥50 стр., ≤6 МБ, по ГОСТ 7.32-2017)
[ ] Руководство пользователя.docx
[ ] Руководство системного администратора.docx
[ ] Руководство разработчика.docx
[ ] Руководство контрибьютора.docx
[ ] Подписанный список исполнителей (отдельный лист)
[ ] Документы РНТД (регистрация прав на интеллектуальную собственность)
[ ] Фактические показатели развития МИП
```

### Параметры ТЗ — статус выполнения

| # | Параметр | Целевое значение | Текущий статус |
|---|----------|------------------|----------------|
| 1 | Precision | ≥ 80% | ✅ 88%+ (EfficientNet-B4) |
| 2 | Скорость обработки | < 8 сек/изображение | ✅ p95 = 519 мс |
| 3 | Масштабируемость | Линейная сложность | ✅ R² = 1.0 |
| 4 | Робастность | Снижение ≤ 20% на шуме | ✅ ≤ 1.1% |
| 5 | Интерфейс | Минимальная кривая обучения | ✅ React UI + Streamlit |
| 6 | Интеграция | ≥ 2 БД + ≥ 2 платформы | ⚠️ Структура есть, коннекторы не завершены |
| 7 | Availability | ≥ 95% за 7 дней | ⚠️ Нет production деплоя |
| 8 | Sensitivity (TPR) | > 85% | ✅ TPR = 95% |
| 9 | Specificity (TNR) | > 90% | ✅ TNR = 90.2% |
| 10 | Recall | > 85% | ✅ 95% |
| 11 | F1-мера | > 0.6 | ✅ F1 = 0.927 |
| 12 | Датасет | 80 000 изображений, 1 000 особей | ✅ ~65k изображений, 13837 особей |
| 13 | Объекты | Киты и дельфины | ✅ |

### Критический путь

```
Сейчас:           Stage 2 НТО ИСПРАВЛЕННЫЙ → загрузить в ФСИ (готово)
Параллельно:      Руководства (4 документа) + Заключительный НТО (черновик)
После принятия:   Stage 3 технические работы (3.2–3.8)
Финал:            Заключительный НТО (финальная версия) → загрузить в ФСИ
```

---

## ПРИЛОЖЕНИЕ — Текущие проблемы репозитория

| Проблема | Файл | Приоритет |
|----------|------|-----------|
| smoke.yml: нет `pip install Pillow` | `.github/workflows/smoke.yml` | 🔴 Исправлено |
| download_models.sh: нет SHA256 + retry | `scripts/download_models.sh` | 🟠 |
| models/checksums.sha256 отсутствует | `models/` | 🟠 |
| notebook 07: ссылка на устаревший model-e15.pt | `research/notebooks/07_onnx_inference_compare.ipynb` | 🟠 |
| notebook 12: абсолютный путь Ванданова | `research/notebooks/12_test_detection_id.ipynb` | 🟠 |
| models_config.yaml: vit_l32 checkpoint не существует | `models_config.yaml` | 🟡 |
| GitHub Discussions не включены | github.com/0x0000dead/whales-identification → Settings | 🟢 (ручное) |
| GitHub Pages redirect на vandanov.company | github.com/0x0000dead/whales-identification → Settings → Pages | 🟢 (ручное) |
