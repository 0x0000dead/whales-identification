# Указатель ноутбуков и артефактов по КП ТЗ

Данный документ закрывает замечание ФСИ: «Требуются ссылки на конкретные ipynb-файлы для каждой работы КП (не ссылка на репозиторий в целом)».

**Репозиторий:** https://github.com/0x0000dead/whales-identification  
**Ветка:** `main`  
**Дата актуализации:** 2026-04-15

---

## Условные обозначения

| Сокращение | Значение |
|------------|----------|
| КП | Ключевой показатель / конкретная работа по ТЗ |
| ipynb | Jupyter notebook |
| py | Python-модуль |
| yml | YAML-конфигурация |

---

## КП 1: Настройка инфраструктуры и CI/CD (Балцат К.И.)

### КП 1.1 — Настройка репозитория для автоматической проверки кода

| Артефакт | Ссылка на GitHub | Назначение |
|----------|-----------------|------------|
| `.github/workflows/ci.yml` | [ci.yml](https://github.com/0x0000dead/whales-identification/blob/main/.github/workflows/ci.yml) | 6-стадийный CI/CD: lint → security → test → docker → trivy → status |
| `.pre-commit-config.yaml` | [.pre-commit-config.yaml](https://github.com/0x0000dead/whales-identification/blob/main/.pre-commit-config.yaml) | 20 хуков: black, flake8, mypy, bandit, isort, nbstripout, interrogate, prettier |
| `.github/labeler.yml` | [labeler.yml](https://github.com/0x0000dead/whales-identification/blob/main/.github/labeler.yml) | Автоматическая расстановка меток на PR |

**Как проверить:** `cd whales_be_service && poetry run pre-commit run --all-files`

### КП 1.2 — Тестирование пилотных прототипов детекции

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/01_backfin-detection-with-yolov5.ipynb` | [01_yolov5.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/01_backfin-detection-with-yolov5.ipynb) | Детекция плавников китов через YOLOv5: аннотации bbox, mAP@0.5 |
| `research/notebooks/01_cetacean-yolov8-train-and-predict.ipynb` | [01_yolov8.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/01_cetacean-yolov8-train-and-predict.ipynb) | Тренировка и предсказание YOLOv8 на датасете спинных плавников |
| `data/backfin_annotations.csv` | [backfin_annotations.csv](https://github.com/0x0000dead/whales-identification/blob/main/data/backfin_annotations.csv) | **5 201 bounding box** — собственная разметка команды (Тарасов А.А.) |

---

## КП 2: Разработка и исследование ML-моделей (Ванданов С.А., Серов А.И.)

### КП 2.1 — Vision Transformer и EfficientNet: тренировка и инференс

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/02_ViT_train_effiecientnet.ipynb` | [02_train.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/02_ViT_train_effiecientnet.ipynb) | Тренировка ViT + EfficientNet на Happy Whale, метрики train/val |
| `research/notebooks/02_ViT_inference_efficientnet.ipynb` | [02_infer.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/02_ViT_inference_efficientnet.ipynb) | Инференс ViT+EfficientNet, top-1/top-5 accuracy, confusion matrix |
| `research/notebooks/03_efficientnet_experiments.ipynb` | [03_effnet.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/03_efficientnet_experiments.ipynb) | Сравнение EfficientNet B0–B5: точность, время инференса, размер модели |

### КП 2.2 — ResNet и Swin Transformer: сравнительный анализ

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/04_resnet_classification_experiments.ipynb` | [04_resnet.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/04_resnet_classification_experiments.ipynb) | ResNet-54/101: Precision/Recall/F1, confusion matrix по 30 видам |
| `research/notebooks/05_swinT_experiments.ipynb` | [05_swinT.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/05_swinT_experiments.ipynb) | Swin Transformer: точность 90%, скорость ~2.2s, параметры 87M |

### КП 2.3 — Бинарная классификация «кит/не-кит» (CLIP anti-fraud gate)

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/06_benchmark_binary.ipynb` | [06_binary.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/06_benchmark_binary.ipynb) | **TPR=0.95, TNR=0.902**, ROC-AUC=0.984, F1=0.927 на 202 изображениях |
| `research/notebooks/06_benchmark_multiclass.ipynb` | [06_multiclass.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/06_benchmark_multiclass.ipynb) | Мультиклассовый бенчмарк EfficientNet-B4 ArcFace: top-1, top-5 accuracy |
| `configs/anti_fraud_threshold.yaml` | [anti_fraud_threshold.yaml](https://github.com/0x0000dead/whales-identification/blob/main/configs/anti_fraud_threshold.yaml) | Порог CLIP = 0.52 (откалиброван на data/test_split/) |

**Подтверждённые метрики (Серов А.И., 15.04.2026):**

```
Anti-fraud gate:  TPR=0.9500  TNR=0.9020  Precision=0.9050  F1=0.9270
Идентификация:    top-1=0.22  top-5=0.25  (13 837 классов, ArcFace s=30 m=0.5)
Латентность:      p50=484ms   p95=519ms   p99=597ms  (бюджет ТЗ: 8000ms)
```

### КП 2.4 — ONNX экспорт для кросс-платформенного инференса

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/07_onnx_inference_compare.ipynb` | [07_onnx.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/07_onnx_inference_compare.ipynb) | Экспорт ViT → ONNX, сравнение скорости PyTorch vs ONNX Runtime |

> **Примечание:** Ноутбук использует legacy ViT модель (`efficientnet_b4_512_fold0.ckpt`). Скачать с Yandex Disk: https://disk.yandex.ru/d/GshqU9o6nNz7ZA

### КП 2.5 — Сравнительная таблица всех архитектур ИНС

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/08_benchmark_all_compare.ipynb` | [08_compare.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/08_benchmark_all_compare.ipynb) | **Полная таблица**: ResNet-54 → ResNet-101 → SwinT → EffB0–B5 → ViT-B16 → ViT-L32 |

**Сводная таблица архитектур (из ноутбука 08):**

| Архитектура | Precision | Скорость CPU (с) | Параметры | Статус |
|-------------|-----------|------------------|-----------|--------|
| ResNet-54 | 82% | ~0.8 | 25M | Базовый |
| ResNet-101 | 85% | ~1.2 | 45M | Baseline |
| Swin-T | 90% | ~2.2 | 87M | Production |
| EfficientNet-B0 | 88% | ~1.0 | 8M | Fast |
| EfficientNet-B4 ArcFace | 91% | ~1.8 | 19M | **Production** |
| ViT-B/16 | 91% | ~2.0 | 86M | Production |
| **ViT-L/32** | **93%** | ~3.5 | 307M | **Best** |

---

## КП 3: Разработка продуктовой системы

### КП 3.1 — Аугментация данных (Тарасов А.А.)

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/09_augmentation_example.ipynb` | [09_augmentation.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/09_augmentation_example.ipynb) | Примеры аугментации: flip, rotate, noise, blur, CLAHE, elastic |
| `whales_identify/dataset.py` | [dataset.py](https://github.com/0x0000dead/whales-identification/blob/main/whales_identify/dataset.py) | `CetaceanDataset` с Albumentations pipeline (12 трансформаций) |
| `whales_identify/filter_processor.py` | [filter_processor.py](https://github.com/0x0000dead/whales-identification/blob/main/whales_identify/filter_processor.py) | Лапласиан-фильтр для отбора чётких изображений (σ > порог) |

### КП 3.2 — Background removal и визуализация

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/10_remove_bg_example.ipynb` | [10_remove_bg.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/10_remove_bg_example.ipynb) | rembg: удаление фона, base64 PNG-маска для API |

### КП 3.3 — Data Stream и видеопоток (Тарасов А.А.)

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/11_data_stram_cv_video.ipynb` | [11_data_stream.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/11_data_stram_cv_video.ipynb) | Data Stream: покадровая обработка видеофайлов через OpenCV |
| `whales_be_service/src/whales_be_service/monitoring/drift.py` | [drift.py](https://github.com/0x0000dead/whales-identification/blob/main/whales_be_service/src/whales_be_service/monitoring/drift.py) | Rolling-window drift detection (окно 1000 сэмплов, порог ±10%) |

### КП 3.4 — Интеграционное тестирование идентификации (Серов А.И.)

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `research/notebooks/12_test_detection_id.ipynb` | [12_test.ipynb](https://github.com/0x0000dead/whales-identification/blob/main/research/notebooks/12_test_detection_id.ipynb) | Сквозной тест детекции + идентификации на тестовой выборке |
| `data/test_split/` | [test_split/](https://github.com/0x0000dead/whales-identification/tree/main/data/test_split) | 100 cetacean + 102 non-cetacean изображений с manifest.csv |
| `whales_be_service/tests/` | [tests/](https://github.com/0x0000dead/whales-identification/tree/main/whales_be_service/tests) | 88 unit-тестов pytest (coverage > 80%) |

### КП 3.5 — Backend API (Ванданов С.А.)

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `whales_be_service/src/whales_be_service/main.py` | [main.py](https://github.com/0x0000dead/whales-identification/blob/main/whales_be_service/src/whales_be_service/main.py) | FastAPI: `/v1/predict-single`, `/v1/predict-batch`, `/metrics`, `/health` |
| `whales_be_service/src/whales_be_service/inference/` | [inference/](https://github.com/0x0000dead/whales-identification/tree/main/whales_be_service/src/whales_be_service/inference) | Pipeline: CLIP gate → EfficientNet-B4 ArcFace → rembg mask |

### КП 3.6 — Frontend UI (Тарасов А.А.)

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `frontend/src/` | [frontend/src/](https://github.com/0x0000dead/whales-identification/tree/main/frontend/src) | React 18 + TypeScript: upload, ConfidenceGauge, RejectionCard |

### КП 3.7 — CI/CD и MLOps (Балцат К.И.)

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `.github/workflows/ci.yml` | [ci.yml](https://github.com/0x0000dead/whales-identification/blob/main/.github/workflows/ci.yml) | 6-стадийный GitHub Actions (lint, security, test, docker, trivy, status) |
| `.github/workflows/smoke.yml` | [smoke.yml](https://github.com/0x0000dead/whales-identification/blob/main/.github/workflows/smoke.yml) | End-to-end smoke test против реального API |
| `.github/workflows/metrics.yml` | [metrics.yml](https://github.com/0x0000dead/whales-identification/blob/main/.github/workflows/metrics.yml) | Автоматический расчёт TPR/TNR на data/test_split/ |
| `.gitlab-ci.yml` | [.gitlab-ci.yml](https://github.com/0x0000dead/whales-identification/blob/main/.gitlab-ci.yml) | GitLab CI: 6 стадий (build → quality → test → artifacts → publish → deploy) |

### КП 3.8 — Системный анализ и классификация (Серов А.И.)

| Артефакт | Ссылка на GitHub | Описание |
|----------|-----------------|----------|
| `whales_be_service/src/whales_be_service/inference/identification.py` | [identification.py](https://github.com/0x0000dead/whales-identification/blob/main/whales_be_service/src/whales_be_service/inference/identification.py) | EfficientNet-B4 ArcFace, fallback chain: effb4 → ViT → ResNet |
| `models/registry.json` | [registry.json](https://github.com/0x0000dead/whales-identification/blob/main/models/registry.json) | Model Registry: версии, метрики, хэши чекпоинтов |
| `MODEL_CARD.md` | [MODEL_CARD.md](https://github.com/0x0000dead/whales-identification/blob/main/MODEL_CARD.md) | Карточка модели: обучение, метрики, ограничения, примеры |
| `docs/GRANT_DELIVERABLES.md` | [GRANT_DELIVERABLES.md](https://github.com/0x0000dead/whales-identification/blob/main/docs/GRANT_DELIVERABLES.md) | Полный маппинг ТЗ§ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ → измеренные значения |

---

## Ссылки на отчёты с метриками

| Файл | Содержание |
|------|-----------|
| `reports/metrics_latest.json` | TPR, TNR, Precision, F1, ROC-AUC, top-1, top-5, latency p50/p95/p99 |
| `reports/METRICS.md` | Текстовый отчёт с интерпретацией результатов |
| `reports/SCALABILITY.md` | R²=0.9982 (линейная сложность по ТЗ §3) |
| `reports/NOISE_ROBUSTNESS.md` | Деградация ≤6.9% при шуме (ТЗ §4: ≤20%) |
| `reports/scalability_latest.json` | JSON-данные бенчмарка масштабируемости |
| `reports/noise_robustness.json` | JSON-данные бенчмарка устойчивости к шуму |
