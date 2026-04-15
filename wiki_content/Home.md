# Welcome to Whales Identification Wiki

![CI/CD](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml/badge.svg)
![Docs](https://github.com/0x0000dead/whales-identification/actions/workflows/deploy-docs.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🐋 О проекте

**Whales Identification (EcoMarineAI)** — библиотека машинного обучения для автоматической детекции и идентификации морских млекопитающих (китов и дельфинов) по аэрофотосъёмке.

### Ключевые возможности

- ✅ **13 837 индивидуальных особей** (15 587-слотовая ArcFace голова, 1 750 резерв)
- ✅ **30 видов** китов и дельфинов (см. `whales_be_service/src/whales_be_service/resources/species_map.csv`)
- ✅ **EfficientNet-B4 + ArcFace** как production identification model
- ✅ **OpenCLIP ViT-B/32** anti-fraud gate (Precision 0.9048 на бинарной задаче)
- ✅ **REST API** с batch processing + webhook + export
- ✅ **Docker Compose** для быстрого развёртывания (prod: k8s manifests в `k8s/`)
- ✅ **Metric Learning** (ArcFace) для масштабируемости
- ✅ **Background removal** с rembg

---

## 📚 Навигация по Wiki

### Быстрый старт

- **[Installation](Installation)** - Пошаговая установка и настройка
- **[Usage](Usage)** - Примеры использования API, Streamlit, notebooks
- **[FAQ](FAQ)** - Часто задаваемые вопросы и решения проблем

### Разработка

- **[Architecture](Architecture)** - Архитектура проекта и компоненты
- **[Model Cards](Model-Cards)** - Подробные характеристики моделей
- **[API Reference](API-Reference)** - Полная документация API с примерами
- **[Testing](Testing)** - Запуск тестов и coverage requirements
- **[Contributing](Contributing)** - Git workflow, code style, pre-commit hooks

### Внешние ресурсы

- **[GitHub Repository](https://github.com/0x0000dead/whales-identification)** - Исходный код
- **[Hugging Face](https://huggingface.co/baltsat/Whales-Identification)** - Обученные модели
- **[Yandex Disk](https://disk.yandex.ru/d/GshqU9o6nNz7ZA)** - Альтернативное хранилище моделей

---

## 🚀 Быстрый старт за 5 минут

```bash
# 1. Клонировать репозиторий
git clone https://github.com/0x0000dead/whales-identification.git
cd whales-identification

# 2. Установить Hugging Face CLI
pip install huggingface_hub==0.20.3

# 3. Загрузить модели
./scripts/download_models.sh

# 4. Запустить полный стек
docker compose up --build
```

**Сервисы:**

- Backend API: http://localhost:8000
- Frontend UI: http://localhost:8080
- API Docs: http://localhost:8000/docs

---

## 📊 Характеристики production-модели

Все числа ниже — воспроизводимые результаты `python scripts/compute_metrics.py`
на выборке `data/test_split/manifest.csv` (202 изображения, 100/102 split).
Не угадывайте — запустите скрипт и сверьте с `reports/metrics_latest.json`.

| Метрика | EfficientNet-B4 ArcFace | ТЗ |
|---------|-------------------------|-----|
| Anti-fraud Precision (бинарная) | **0.9048** | §Параметр 8/9/11 |
| Anti-fraud Recall / F1 | 0.950 / 0.9268 | §Параметр 8, 11 |
| Species precision (high-conf) | 0.5294 | §Параметр 1 (см. `reports/METRICS.md` для обоснования interpretability) |
| Latency p95 (CPU) | 298.87 мс | §Параметр 2 (≤ 8 с) ✓ |
| Latency p99 (CPU) | 416.73 мс | §Параметр 2 ✓ |
| Масштабируемость (R²) | 1.000 | §Параметр 3 (линейная) ✓ |
| Устойчивость к шуму (drop) | −1.1 % | §Параметр 4 (≤ 20 %) ✓ |

**Подробности:** [Model Cards](Model-Cards), `DOCS/PERFORMANCE_REPORT.md`,
`reports/METRICS.md`. Сравнение альтернативных архитектур (ResNet, Swin,
ViT-L/32) см. в `research/notebooks/08_benchmark_all_compare.ipynb`.

---

## 🛠️ Технологический стек

### Backend

- **Framework:** FastAPI (Python 3.11.6)
- **ML:** PyTorch 2.4.1, TIMM, Vision Transformers
- **Inference:** ArcFace metric learning, GeM pooling
- **Package Manager:** Poetry

### Frontend

- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Visualization:** Recharts

### DevOps

- **CI/CD:** GitHub Actions (6-stage pipeline)
- **Quality:** 20 pre-commit hooks
- **Containerization:** Docker + Docker Compose
- **Documentation:** GitHub Pages + Wiki

---

## ⚠️ Важные ограничения

### Лицензионные ограничения

Проект использует **трёхуровневую модель лицензирования**:

1. **Исходный код:** MIT License (свободное использование)
2. **Обученные модели:** CC-BY-NC-4.0 (наследует от тренировочных данных)
3. **Датасеты:** CC-BY-NC-4.0 + Government Research-Only

❌ **Коммерческое использование запрещено** из-за:

- Happy Whale (CC-BY-NC-4.0)
- Данные Минприроды РФ (research-only)
- ImageNet pretrained weights (non-commercial)

**Подробности:**

- [LICENSE](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE) - Исходный код
- [LICENSE_MODELS.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_MODELS.md) - Модели
- [LICENSE_DATA.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_DATA.md) - Датасеты
- [LICENSES_ANALYSIS.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSES_ANALYSIS.md) - Анализ зависимостей

> **⚠️ Disclaimer:** Информация о лицензиях предоставлена для ознакомительных целей и не является юридической консультацией. Для коммерческого использования рекомендуется консультация с квалифицированным юристом.

---

## 👥 Команда разработки

- **Baltsat Konstantin** - ML Engineering, CI/CD
- **Tarasov Artem** - Backend Development
- **Vandanov Sergey** - Frontend Development
- **Serov Alexandr** - ML Research

---

## 📞 Поддержка

- **Issues:** [Report a bug or ask a question](https://github.com/0x0000dead/whales-identification/issues)

---

## 📈 CI/CD Pipeline

Проект использует comprehensive GitHub Actions pipeline:

- ✅ **Lint:** black, flake8, isort, mypy
- ✅ **Security:** bandit, safety, trivy
- ✅ **Tests:** pytest with coverage >80%
- ✅ **Docker:** Build, integration tests, health checks
- ✅ **Docs:** Auto-deploy to GitHub Pages

**Статус:** [![CI/CD](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml/badge.svg)](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml)

---

**Последнее обновление:** 1 сентября 2025
**Версия:** 0.1.0
