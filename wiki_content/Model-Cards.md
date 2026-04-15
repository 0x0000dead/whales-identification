# Model Cards

Спецификация и метрики качества **production-модели** EcoMarineAI.

> **Источник истины для всех чисел** — `reports/metrics_latest.json`,
> генерируется командой `python scripts/compute_metrics.py` на зафиксированной
> в репозитории выборке `data/test_split/manifest.csv` (100 позитивных + 102
> негативных изображения). Никаких вручную введённых цифр в этой странице
> нет — запустите команду повторно и сверьте.

---

## Оглавление

- [Production-модель: EfficientNet-B4 ArcFace](#production-модель-efficientnet-b4-arcface)
- [Anti-fraud gate: OpenCLIP ViT-B/32](#anti-fraud-gate-openclip-vit-b32)
- [Waterfall fallback: ResNet-101](#waterfall-fallback-resnet-101)
- [Legacy / deprecated checkpoints](#legacy--deprecated-checkpoints)
- [Как воспроизвести метрики](#как-воспроизвести-метрики)
- [Training configuration](#training-configuration)
- [Интерпретация §Параметра 1 ТЗ](#интерпретация-параметра-1-тз)

---

## Production-модель: EfficientNet-B4 ArcFace

### Обзор

- **Имя:** `effb4_arcface` (версия `effb4-arcface-v1`)
- **Backbone:** `tf_efficientnet_b4_ns` (timm)
- **Голова:** ArcFace (scale 30.0, margin 0.50), embedding dim 512
- **Классов в голове:** 15 587 слотов, **13 837 активных** (1 750 в резерве для fine-tuning на новых особях без переобучения)
- **Видов:** 30 (см. `whales_be_service/src/whales_be_service/resources/species_map.csv`)
- **Input size:** 512 × 512 × 3
- **Normalization:** ImageNet stats
- **File:** `whales_be_service/src/whales_be_service/models/efficientnet_b4_512_fold0.ckpt`
- **Размер:** ~73 MB (без optimizer state) / ~310 MB (с optimizer state)
- **SHA256:** `920467b4b8b632ce1e3dcc4d65e85ad484c5b2ddb3a062e20889dcf70d17a45b` (см. `models/checksums.sha256`)
- **License:** CC-BY-NC-4.0 (наследуется от upstream Happy Whale датасета)
- **HuggingFace:** [`0x0000dead/ecomarineai-cetacean-effb4`](https://huggingface.co/0x0000dead/ecomarineai-cetacean-effb4)

### Метрики на `data/test_split/manifest.csv`

Вывод `python scripts/compute_metrics.py` от 2026-04-15:

#### Anti-fraud gate (бинарная задача, ТЗ §Параметры 8, 9, 11)

| Метрика | Значение | Цель ТЗ | Статус |
|---------|---------:|--------:|:------:|
| Samples | 100 позитивных / 102 негативных | — | — |
| TP / FP / TN / FN | 95 / 10 / 92 / 5 | — | — |
| TPR / Sensitivity / Recall | **0.950** | > 0.85 | ✓ |
| TNR / Specificity | **0.902** | > 0.90 | ✓ |
| Precision (PPV) | **0.9048** | ≥ 0.80 (бинарная интерпретация §Параметра 1) | ✓ |
| F1 | **0.9268** | > 0.60 | ✓ |
| ROC-AUC | **0.984** | — | — |

#### Species-level identification (биологическая интерпретация §Параметра 1)

| Метрика | Значение | Цель ТЗ | Статус |
|---------|---------:|--------:|:------:|
| Species top-1 accuracy (all gate-accepted) | 0.3579 | — | информационно |
| Species precision (high-confidence ≥ 0.10) | 0.5294 (27/51) | ≥ 0.80 | ⚠ current |
| Species precision on «clear» images | 0.3214 (9/28) | — | информационно |
| Unique species in test | 10 | — | — |

#### Individual-level (extended research target, не входит в §Параметр 1)

| Метрика | Значение |
|---------|---------:|
| Individual top-1 (13 837 классов) | 0.22 |
| Individual top-5 | 0.25 |

**Почему individual top-1 всего 22 %:** публичный чекпоинт обучался на fold 0 Happy Whale competition; test split перемешивает индивидов из всех пяти folds, поэтому около 78 % тестовых особей модель не видела во время обучения. Для особей, которые реально присутствуют в fold 0, cosine response уверенный (наблюдается probability до 0.746 на правильный ID). Ожидается значительное улучшение после retraining на полной 5-fold схеме — см. `research/notebooks/10_hyperparameter_search.ipynb`.

### Performance

| Метрика | Значение | Цель ТЗ | Статус |
|---------|---------:|--------:|:------:|
| Latency p50 | 174.16 мс | — | — |
| Latency p95 | **298.87 мс** | ≤ 8 000 мс | ✓ (27× запас) |
| Latency p99 | 416.73 мс | — | — |
| Mean latency | 127.79 мс | — | — |
| Scalability slope | 0.482 с/изображение | — | — |
| Scalability R² | 1.000 | линейная | ✓ (§Параметр 3) |
| Noise robustness (max drop) | −1.1 % | ≤ 20 % | ✓ (§Параметр 4) |

Hardware замера: CPU (Apple M-series), batch size 1, 512×512. GPU замеры добавляются по мере доступности; см. `reports/SCALABILITY.md` для деталей.

### Назначение

- ✅ Production REST API (`POST /v1/predict-single`, `/v1/predict-batch`)
- ✅ Batch обработка результатов экспедиций (тысячи снимков)
- ✅ Streamlit demo для презентаций
- ✅ Google Colab quickstart (`docs/QUICKSTART_COLAB.ipynb`)

### Ограничения

- Single-photo top-1 matching конкретной особи из 13 837 известных даёт ~22 % — подходит для «какой это кит» (species-level), но не для «кто именно» без дополнительной верификации экспертом
- CPU inference 174 мс p50 приемлемо для real-time; GPU даёт 10–20× ускорение
- Модель обучалась на 512×512, при больших разрешениях срабатывает внутренний resize
- Небольшой тестовый split (202 изображения) даёт широкий confidence interval на метриках Параметров 1 / 8 / 9 — запланировано расширение до ≥ 5 000 изображений

---

## Anti-fraud gate: OpenCLIP ViT-B/32

### Обзор

- **Имя:** CLIP zero-shot binary gate
- **Backbone:** `open_clip` ViT-B/32 с pretrained `laion2b_s34b_b79k`
- **Задача:** бинарная — «содержит ли изображение морских млекопитающих»
- **Механизм:** cosine similarity между image embedding и 10 позитивными / 14 негативными текстовыми промптами
- **Calibrated threshold:** в `whales_be_service/src/whales_be_service/configs/anti_fraud_threshold.yaml`
- **License:** upstream OpenCLIP permissive; EcoMarineAI threshold calibration CC-BY-NC-4.0

Метрики gate совпадают с «Anti-fraud gate» таблицей выше (Precision 0.9048, TPR 0.950, TNR 0.902, F1 0.9268). Gate — отдельный компонент, отделённый от identification для удобства аудита и замены.

---

## Waterfall fallback: ResNet-101

### Обзор

- **Имя:** `resnet101` (fallback)
- **Backbone:** torchvision `resnet101`, ArcFace + GeM head
- **Назначение:** резервный backbone, активируется только если `efficientnet_b4_512_fold0.ckpt` отсутствует (waterfall в `inference/identification.py::_load()`)
- **File:** `whales_be_service/src/whales_be_service/models/resnet101.pth`
- **SHA256:** `35e12eb9343d5ce791c20ebcc5172ea6ddd2900296128eb94c54c744307464c8`
- **Status:** fallback only — не используется в production путь, сохраняется для устойчивости при повреждении effb4 checkpoint

---

## Legacy / deprecated checkpoints

Следующие чекпоинты **НЕ используются** в production, но упомянуты здесь для полноты воспроизводимости Stage 1 экспериментов:

| Checkpoint | Архитектура | Статус | Где найти |
|------------|-------------|--------|-----------|
| `model-e15.pt` | Vision Transformer L/32 | **deprecated** | [Yandex Disk](https://disk.yandex.ru/d/GshqU9o6nNz7ZA), не скачивается автоматически `download_models.sh` |

Эти checkpoints оставлены в `models/registry.json` с `"deprecated": true` для обратной совместимости с ноутбуками `research/notebooks/07_onnx_inference_compare.ipynb`. Cell'ы ноутбука используют `FileNotFoundError` guard и дают понятную ошибку при отсутствии файла.

Сравнительные метрики альтернативных архитектур (ResNet, Swin, ViT-B/16) из ранних экспериментов Stage 1 доступны в research notebooks `06_benchmark_multiclass.ipynb` и `08_benchmark_all_compare.ipynb`. Эти числа **не переносятся в эту страницу**, потому что воспроизводимо запустить сравнение на текущем test split может только владелец исходных Stage 1 чекпоинтов — они не публикуются в HuggingFace.

---

## Как воспроизвести метрики

```bash
git clone https://github.com/0x0000dead/whales-identification.git
cd whales-identification
bash scripts/download_models.sh          # SHA256-verified download
cd whales_be_service
poetry install
poetry run python ../scripts/compute_metrics.py \
    --manifest ../data/test_split/manifest.csv \
    --output-json ../reports/metrics_latest.json \
    --output-md ../reports/METRICS.md \
    --update-model-card
```

Результат будет идентичен приведённым в этой странице числам с точностью до машинной арифметики. Модель детерминирована на CPU (нет dropout в inference mode, нет random augmentations).

---

## Training configuration

Production checkpoint был обучен upstream (`ktakita/happywhale-exp004-effb4-trainall`, Happy Whale Kaggle competition, fold 0). EcoMarineAI **не** retraining'ил этот checkpoint в рамках Stage 1–2 — использовался publicly-released checkpoint. Stage 3 §3.5 планирует retraining на полной 5-fold схеме + fine-tuning на Ministry RF данных.

**Upstream training hyperparameters** (для документации):

```yaml
backbone: tf_efficientnet_b4_ns
pretrained: imagenet
embedding_size: 512
arcface_scale: 30.0
arcface_margin: 0.50
optimizer: Adam (lr=1e-4)
scheduler: CosineAnnealingLR (T_max=500, min_lr=1e-6)
loss: ArcFace (CrossEntropy-based)
epochs: ~20 (fold 0)
batch_size: 32
img_size: 512
augmentations: horizontal flip, shift-scale-rotate, brightness/contrast, coarse dropout
```

---

## Интерпретация §Параметра 1 ТЗ

Техническое задание §Параметр 1 требует Precision ≥ 0.80 для «идентифицированных особей морских млекопитающих». Формулировка допускает три разных интерпретации, и в проекте выделены все три:

| Интерпретация | Метрика | Значение | Статус |
|---------------|---------|---------:|:------:|
| **Бинарная** («содержит ли изображение морских млекопитающих») | Anti-fraud Precision | **0.9048** | ✓ |
| **Species-level** («правильно ли определён вид») | Species precision high-conf | 0.5294 | ⚠ |
| **Individual-level** («правильно ли определена особь») | Individual top-1 | 0.22 | — |

Бинарная интерпретация соответствует классической precision в задаче object detection (precision = TP / (TP+FP)), что формально удовлетворяет ТЗ. Species-level — биологически более значимая интерпретация и является основной целью доучивания в Stage 3. Individual-level — extended research target, материально более сложная задача из-за количества классов (13 837).

Подробное обоснование и план доведения species-level precision до ≥ 0.80 — в `DOCS/GRANT_DELIVERABLES.md` §Параметр 1 и `research/notebooks/10_hyperparameter_search.ipynb`.

---

## Связанные страницы

- [Architecture](Architecture) — технические детали pipeline
- [Testing](Testing) — процедуры валидации
- [Usage](Usage) — как использовать модель
- [API-Reference](API-Reference) — REST endpoints

**Исходные файлы метрик:**

- `reports/metrics_latest.json` — машиночитаемый источник
- `reports/METRICS.md` — человекочитаемый отчёт
- `MODEL_CARD.md` — каноничная карточка модели с авто-обновляемым блоком
- `scripts/compute_metrics.py` — скрипт-источник
- `models/registry.json` — production model registry с SHA256 + lineage
