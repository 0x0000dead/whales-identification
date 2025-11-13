# Whales Identification - Documentation

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11.6-blue.svg)
![CI/CD](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml/badge.svg)

## 🐋 О проекте

**Whales Identification** — ML-библиотека для автоматической детекции и идентификации морских млекопитающих (китов и дельфинов) по аэрофотосъёмке. Проект использует глубокое обучение (PyTorch) с архитектурами Vision Transformers, EfficientNet и ResNet.

---

## 📚 Документация

### Основные разделы

- **[GitHub Repository](https://github.com/0x0000dead/whales-identification)** - Исходный код проекта
- **[GitHub Wiki](https://github.com/0x0000dead/whales-identification/wiki)** - Полная документация с примерами
- **[API Documentation](https://github.com/0x0000dead/whales-identification/wiki/API-Reference)** - REST API endpoints и примеры
- **[Installation Guide](https://github.com/0x0000dead/whales-identification/wiki/Installation)** - Установка и настройка

### Технические гайды

- **[Pre-commit Hooks Guide](PRE_COMMIT_GUIDE.md)** - Настройка и использование pre-commit hooks
- **[Architecture](https://github.com/0x0000dead/whales-identification/wiki/Architecture)** - Архитектура проекта
- **[Model Cards](https://github.com/0x0000dead/whales-identification/wiki/Model-Cards)** - Описание моделей и метрик
- **[Testing Guide](https://github.com/0x0000dead/whales-identification/wiki/Testing)** - Запуск тестов и coverage

---

## 📄 Лицензирование

Проект использует трёхуровневую модель лицензирования:

### 1. Исходный код
- **Лицензия:** MIT License
- **Файл:** [LICENSE](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE)
- **Права:** Свободное использование, модификация, коммерческое применение

### 2. Обученные модели
- **Лицензия:** Apache 2.0 с ограничениями
- **Файл:** [LICENSE_MODELS.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_MODELS.md)
- **⚠️ Ограничения:**
  - ❌ Коммерческое использование запрещено (из-за HappyWhale CC-BY-NC-4.0)
  - ❌ Только исследовательские цели (данные Минприроды РФ)
  - ❌ Pretrained модели ImageNet (non-commercial terms)

### 3. Датасеты
- **Лицензия:** CC-BY-NC-4.0 (HappyWhale) + Government Research-Only (Минприроды РФ)
- **Файл:** [LICENSE_DATA.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_DATA.md)
- **Ограничения:** Только некоммерческое исследовательское использование

### 4. Анализ зависимостей
- **Файл:** [LICENSES_ANALYSIS.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSES_ANALYSIS.md)
- **Совместимость:** 99.4% (158/159 зависимостей MIT-совместимы)

---

## 🔧 CI/CD & Автоматизация

### GitHub Actions Workflows

- **[CI/CD Pipeline](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml)**
  - Linting (black, flake8, mypy)
  - Security scanning (bandit, trivy)
  - Testing with coverage
  - Docker build and integration tests

- **[Documentation Deployment](https://github.com/0x0000dead/whales-identification/actions/workflows/deploy-docs.yml)**
  - Автоматический деплой docs/ на GitHub Pages
  - Триггер: push в main с изменениями в docs/

### Pre-commit Hooks

Проект использует 20 pre-commit hooks для обеспечения качества кода:

| Категория | Хуки | Автофикс |
|-----------|------|----------|
| **Форматирование** | black, isort, prettier | ✅ Да |
| **Линтинг** | flake8 | ❌ Нет |
| **Типизация** | mypy | ❌ Нет |
| **Безопасность** | bandit | ❌ Нет |
| **Jupyter** | nbstripout, nbqa-black, nbqa-isort, nbqa-flake8 | ✅ Частично |
| **Базовые проверки** | 8 hooks (trailing-whitespace, check-yaml, etc.) | ✅ Большинство |

**Установка:**
```bash
cd whales_be_service
poetry install
poetry run pre-commit install
```

**Подробная документация:** [PRE_COMMIT_GUIDE.md](PRE_COMMIT_GUIDE.md)

---

## 🚀 Быстрый старт

### Требования

- Python 3.11.6
- Docker & Docker Compose
- Poetry (backend package manager)
- npm (frontend package manager)
- Hugging Face CLI (для загрузки моделей)

### Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/0x0000dead/whales-identification.git
cd whales-identification

# 2. Установить Hugging Face CLI
pip install huggingface-hub[cli]

# 3. Загрузить модели
./scripts/download_models.sh

# 4. Запустить полный стек
docker compose up --build
```

**Сервисы:**
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:8080
- API Docs: http://localhost:8000/docs

**Подробная инструкция:** [GitHub Wiki - Installation](https://github.com/0x0000dead/whales-identification/wiki/Installation)

---

## 📊 Характеристики моделей

| Модель | Precision | Время (с) | Статус |
|--------|-----------|-----------|--------|
| **Vision Transformer L/32** | 93% | ~3.5s | ⭐ Best |
| Vision Transformer B/16 | 91% | ~2.0s | ✅ Production |
| EfficientNet-B5 | 91% | ~1.8s | ✅ Production |
| EfficientNet-B0 | 88% | ~1.0s | ⚡ Fast |
| ResNet-101 | 85% | ~1.2s | ✅ Baseline |

**Подробные характеристики:** [Model Cards](https://github.com/0x0000dead/whales-identification/wiki/Model-Cards)

---

## 🛠️ Технологический стек

### Backend
- **Framework:** FastAPI (Python 3.11.6)
- **ML:** PyTorch 2.4.1, TIMM models
- **Inference:** Vision Transformers, metric learning (ArcFace)
- **Package Manager:** Poetry

### Frontend
- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Charts:** Recharts

### DevOps
- **Containerization:** Docker + Docker Compose
- **CI/CD:** GitHub Actions
- **Code Quality:** 20 pre-commit hooks
- **Documentation:** GitHub Pages, GitHub Wiki

---

## 👥 Команда разработки

- **Baltsat Konstantin** - ML Engineering, CI/CD
- **Tarasov Artem** - Backend Development
- **Vandanov Sergey** - Frontend Development
- **Serov Alexandr** - ML Research

---

## 📞 Контакты и поддержка

- **GitHub Issues:** [Report a bug](https://github.com/0x0000dead/whales-identification/issues)
- **GitHub Discussions:** [Ask a question](https://github.com/0x0000dead/whales-identification/discussions)
- **Email:** konstantin.baltsat@example.com

---

## 📈 Статус проекта

- ✅ Полный CI/CD pipeline с 6 стадиями
- ✅ 20 pre-commit hooks для quality assurance
- ✅ Comprehensive licensing documentation
- ✅ Docker-based deployment
- ✅ REST API с автоматической документацией
- ✅ Batch processing support
- 🚧 GitHub Wiki (in progress)
- 🚧 Performance monitoring (planned)

---

## 📜 Цитирование

Если вы используете этот проект в исследовании, пожалуйста, укажите:

```bibtex
@software{whales_identification_2024,
  author = {Baltsat, Konstantin and Tarasov, Artem and Vandanov, Sergey and Serov, Alexandr},
  title = {Whales Identification: ML Library for Marine Mammal Detection},
  year = {2024},
  url = {https://github.com/0x0000dead/whales-identification}
}
```

---

**Последнее обновление:** 1 сентября 2025
**Версия:** 0.1.0
**GitHub Pages:** https://0x0000dead.github.io/whales-identification/
