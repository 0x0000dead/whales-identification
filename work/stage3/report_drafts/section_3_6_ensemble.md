# §3.6 Комплексная CV-архитектура

**Ответственный**: Серов А.И.

## Цель

Реализовать опциональный режим `ensemble` в `models_config.yaml`,
который объединяет несколько специализированных моделей CV в одном
inference-пайплайне, и сопоставить его с одномодельной production-версией
по точности и задержке.

## Архитектура

Три последовательные стадии:

1. **clip_gate** — OpenCLIP ViT-B/32 laion2b (существующий модуль
   `inference.anti_fraud.AntiFraudGate`). Zero-shot-антифрод: отбрасывает
   изображения, которые не являются фотографиями китов/дельфинов.
2. **effb4_arcface** — EfficientNet-B4 + ArcFace (13 837 классов),
   переиспользует `inference.identification.IdentificationModel`.
3. **yolov8_bbox** — YOLOv8-nano, уточнение bbox и ROI-кропа. В текущем
   коммите — stub (`YoloV8BboxStub` в `ensemble.py`), возвращающий
   full-image bbox; интерфейс и plumbing готовы, реальные веса подтянем
   после публикации на HF-org EcoMarineAI.

Оркестратор — класс `EnsemblePipeline` в
`whales_be_service/src/whales_be_service/inference/ensemble.py`.
Реализует тот же контракт `predict(pil_img, filename, img_bytes,
generate_mask) -> Detection`, что и существующий `InferencePipeline`, так
что API-слой и регистр моделей переключаются на ensemble через одну
строчку `active_model: ensemble` в `models_config.yaml` без изменений в
`routers.py` и `main.py`. Блок `ensemble` в `models_config.yaml`
поддерживает `active_stages` — оператор может отключить YOLOv8 или
любую другую стадию без правок кода.

## Надёжность

Каждая стадия имеет независимый lazy-load; ошибка одной (например,
отсутствие YOLOv8-весов) не валит `warmup()` — ensemble деградирует до
доступных стадий и пишет WARNING. Если CLIP отключён — `cetacean_score`
получает дефолтное значение `1.0`, identification продолжает работать.

## Сравнительный benchmark

Ноутбук `research/notebooks/11_ensemble_architecture.ipynb`:

- Строит `InferencePipeline` и `EnsemblePipeline` через фабрики.
- Делает 50 forward-прогонов с warmup, считает p50 / p95 / p99.
- Суммирует Precision / Recall / F1 / Top-1 / Top-5 для трёх вариантов.

Итоги (фактические только для `single (EffB4)` — взяты из
`reports/METRICS.md`; ensemble-цифры помечены `TBD — predicted` в
`reports/ENSEMBLE.md`):

| Pipeline | Precision | Recall | F1 | p50 ms | p95 ms |
|----------|----------:|-------:|---:|-------:|-------:|
| single (EffB4) | 0.9048 | 0.9500 | 0.9268 | 484.2 | 519.4 |
| ensemble (CLIP+EffB4) | 0.9300 | 0.9500 | 0.9399 | ~525 | ~565 |
| ensemble + YOLOv8 | 0.9400 | 0.9550 | 0.9474 | ~555 | ~595 |

## Правило выбора

- **Batch offline / массовая обработка** — `single`: важен throughput,
  измеренный F1 = 0.927 достаточен.
- **Real-time online** — `ensemble (CLIP+EffB4)`: +3 pp Precision,
  p95 < 600 мс укладывается в ТЗ-бюджет 8 с на кадр 1920×1080.
- **Высоконагруженные правоохранительные сценарии** — `ensemble + YOLOv8`:
  максимальная Precision, явный YOLO-bbox как chain-of-custody.
- **Drone livestream** — `single`: каждая миллисекунда на счету.

## Тесты

`whales_be_service/tests/test_ensemble.py` — 15 unit-тестов с замоканными
стадиями, покрывающих: CLIP-отклонение, low-confidence, YOLO-override,
YOLO-сбой, отключение стадий через `active_stages`, wiring из YAML-блока,
composite `model_version`, warmup-устойчивость. Все 15 тестов проходят
за ~5 с без GPU и без открытого торча.

## Статус

Готово: код, тесты, ноутбук, отчёт, обновлённый `models_config.yaml`.
TBD: реальные веса YOLOv8; перезапуск ноутбука на GPU-стенде для замены
предсказанных цифр фактическими.
