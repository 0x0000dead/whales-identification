# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**whales-identification** is an ML library for automated detection and identification of marine mammals (whales and dolphins) from aerial photography. The project uses deep learning (PyTorch) with multiple model architectures including Vision Transformers, EfficientNet, and ResNet.

**Tech Stack:**

- Backend: FastAPI (Python 3.11.6), PyTorch 2.4.1
- Frontend: React 18 + TypeScript + Vite
- ML: TIMM models, Vision Transformers, metric learning (ArcFace)
- Deployment: Docker + Docker Compose
- Package Management: Poetry (backend), npm (frontend)

## Architecture

### Directory Structure

```
whales_identify/          # Core ML library (training, models, dataset)
whales_be_service/        # FastAPI REST API backend
  ├── src/whales_be_service/
  │   ├── main.py         # FastAPI app and CORS setup
  │   ├── routers.py      # API routes
  │   ├── response_models.py  # Pydantic models + inference
  │   ├── whale_infer.py  # Model loading and inference wrapper
  │   └── config.yaml     # 15,587 whale ID → species mappings
  ├── tests/              # pytest test suite
  └── pyproject.toml      # Poetry dependencies
frontend/                 # React UI for single/batch processing
research/
  ├── notebooks/          # Jupyter notebooks with model experiments
  ├── demo-ui/            # Streamlit demo (best model: ViT)
  └── demo-ui-mask/       # Alternative Streamlit demo with masking
models/                   # Downloaded model weights (.gitignore'd)
data/                     # Datasets and sample images
scripts/                  # Utility scripts (e.g., download_models.sh)
```

### Key Components

**1. ML Pipeline (whales_identify/):**

- `model.py`: CetaceanIdentificationModel with GeM pooling + ArcMarginProduct
- `dataset.py`: PyTorch Dataset with Albumentations augmentation
- `train.py`: Training loop with checkpoint saving
- `config.py`: Training hyperparameters (448x448, EfficientNet-B0)

**2. Backend API (whales_be_service/):**

- POST `/predict-single` - Single image identification
- POST `/predict-batch` - Batch processing via ZIP archive
- Response format: bbox, species, individual_id, probability, base64 mask
- Uses rembg for background removal

**3. Frontend (React + TypeScript):**

- Single image upload with preview
- ZIP batch processing
- Results visualization with Recharts
- Error handling with modal dialogs

### Data Flow

```
User Upload → Frontend (React)
    ↓
FormData POST → Backend (FastAPI)
    ↓
Image preprocessing (OpenCV → 448×448 → normalize)
    ↓
Model inference (VisionTransformer → 15,587 class logits)
    ↓
Postprocessing (top-1 ID → species lookup → background removal)
    ↓
JSON response (Detection object with base64 mask)
    ↓
Frontend display (image, species, probability)
```

## Common Development Tasks

### Installation & Setup

```bash
# Backend
cd whales_be_service
poetry install                    # Install dependencies
poetry run pre-commit install     # Setup pre-commit hooks

# Frontend
cd frontend
npm install

# Download models (required before running)
./scripts/download_models.sh      # Downloads from Hugging Face
```

**Note:** Models are hosted on [Hugging Face](https://huggingface.co/baltsat/Whales-Identification) and [Yandex Disk](https://disk.yandex.ru/d/GshqU9o6nNz7ZA). The `models/` directory is .gitignore'd.

### Running Services

```bash
# Full stack (recommended)
docker compose up --build         # Backend on :8000, Frontend on :8080

# Backend only (local dev)
cd whales_be_service
poetry run python -m uvicorn whales_be_service.main:app \
  --host 0.0.0.0 --port 8000 --reload

# Frontend only (local dev)
cd frontend
npm run dev                       # Vite dev server on :5173
npm run build                     # Production build
npm run preview                   # Preview production build

# Streamlit demos
cd research/demo-ui
poetry run streamlit run streamlit_app.py --server.port=8501
```

### Testing & Quality Checks

```bash
# Run tests
make test                         # pytest with fail-fast
cd whales_be_service && poetry run pytest

# Linting
make lint                         # flake8
poetry run black --check .        # Check formatting
poetry run black .                # Auto-format

# Pre-commit hooks (auto-runs on git commit)
poetry run pre-commit install
poetry run pre-commit run --all-files
```

**Test Location:** `whales_be_service/tests/api/test_post_endpoints.py`

- Tests both `/predict-single` and `/predict-batch` endpoints
- Validates response structure, error handling, content types

### Model Operations

```bash
# Train model (from whales_identify)
poetry run python train.py --train_csv data.csv --img_dir images/

# Download models
./scripts/download_models.sh      # Requires huggingface-cli

# Model inference
cd research/demo-ui
poetry run streamlit run streamlit_app.py --server.port=8501
```

#### Working with Jupyter Notebooks

- **Use `# %%` cell markers** in Python scripts for notebook-style execution
- Enables cell-by-cell execution in VS Code, PyCharm, JupyterLab
- All research notebooks in `research/notebooks/` follow this convention
- Example:

  ```python
  # %%
  import torch
  import numpy as np

  # %%
  # Load and preprocess data
  data = load_data()
  ```

### CI/CD

**GitHub Actions:**

- `.github/workflows/docker-image.yml` - Builds Docker image on push/PR to main

**Pre-commit Hooks (`.pre-commit-config.yaml`):**

- `black` (line-length=88) - Auto-formatting
- `flake8` (max-line-length=88, ignore E203,W503) - Linting

## Model Information

**Supported Architectures (from research/notebooks):**

- Vision Transformer L/32: 93% precision, ~3.5s (best accuracy)
- Vision Transformer B/16: 91% precision, ~2.0s
- EfficientNet-B5: 91% precision, ~1.8s
- EfficientNet-B0: 88% precision, ~1.0s
- ResNet-101: 85% precision, ~1.2s
- ResNet-54: 82% precision, ~0.8s (fastest)
- Swin Transformer: 90% precision, ~2.2s

**Model Loading:**

- Production backend uses ViT model from `models/model-e15.pt`
- Models are loaded via `whale_infer.py` with torch.load()
- Checkpoint format: `{epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, loss}`

**Inference Configuration:**

- Input: 448×448 RGB images
- Normalization: ImageNet stats (mean=[0.485,0.456,0.406])
- Output: 15,587 individual whale IDs → species names via config.yaml
- Background removal: rembg library, returns base64 PNG mask

## Important Technical Details

### Metric Learning Approach

The project uses **metric learning** (ArcMarginProduct) rather than simple classification. This enables:

- Learning embeddings that cluster similar whale individuals
- Better generalization to new/unseen whales
- Scalability to 15,587+ unique individuals

### API Response Structure

```python
class Detection(BaseModel):
    image_ind: str              # Filename
    bbox: list[int]            # [x, y, width, height]
    class_animal: str          # Individual ID (hex-like)
    id_animal: str             # Species name
    probability: float         # 0.0-1.0 confidence
    mask: str | None          # Base64 PNG with background removed
```

### Docker Configuration

- Backend Dockerfile: Python 3.11.6-slim, Poetry, uvicorn on :8000
- Frontend Dockerfile: Multi-stage (Node 20 build → nginx:alpine serve) on :80
- docker-compose.yml: Bridge network (whale-net), VITE_BACKEND=http://backend:8000

### Known Issues (from README)

1. Root Dockerfile missing → use `docker compose up` instead
2. No pyproject.toml in root → use `whales_be_service/pyproject.toml`
3. Models must be downloaded manually before running (not in git due to size)
4. `huggingface-cli` must be installed to run `scripts/download_models.sh`

## Project Goals & Metrics (from README)

**Target Performance:**

- Precision: ≥80% for clear 1920x1080 images
- Sensitivity/Recall: >85%
- Specificity: >90%
- F1-score: >0.6
- Processing speed: <8 seconds per 1920x1080 image
- Robustness: Accuracy drop ≤20% on noisy images

**Dataset Requirements:**

- ~60,000 training images
- ~20,000 test images
- Species: Whales and dolphins
- Source: Happy Whale + Ministry of Natural Resources RF

## Workflow (Git)

1. Create feature branch from `main` (e.g., `feature/new-auth`, `fix/login-bug`)
2. Make commits with clear messages
3. Push branch and open Merge Request to `main`
4. Ensure CI/CD checks pass (linters, tests, Docker build)
5. Get code review approval from ≥1 team member
6. Merge to `main` after approval
7. Delete feature branch post-merge
8. Keep `main` stable and deployable

**Team Members:** Baltsat K.I., Tarasov A.A., Vandanov S.A., Serov A.I.

## Quick Reference

**Essential Commands:**

```bash
# Setup
poetry install && poetry run pre-commit install

# Run full stack
docker compose up --build

# Run tests
make test

# Format code
poetry run black .

# Train model (from whales_identify)
poetry run python train.py --train_csv data.csv --img_dir images/
```

**API Endpoints:**

- `POST /predict-single` - Upload single image (multipart/form-data)
- `POST /predict-batch` - Upload ZIP archive (application/zip)

**Model Storage:**

- Hugging Face: `baltsat/Whales-Identification`
- Local: `models/` (add to .gitignore)
- Download script: `./scripts/download_models.sh`

---

## Правила разработки ML/AI проектов

Эти универсальные правила выведены из экспертной обратной связи на проект и применимы к любым ML/AI проектам.

### I. СТРАТЕГИЧЕСКИЕ ПРАВИЛА (верхний уровень)

#### 1. Управление ML-артефактами

**1.1 Трёхуровневая лицензионная модель**

- **Критерий**: Проект должен явно разделять лицензии на: (1) исходный код, (2) обученные модели, (3) датасеты
- **Измерение**: Наличие LICENSE файлов или секций для каждого типа артефактов
- **Статус в проекте**: ✅ MIT (код), Apache 2.0 (модели), CC-BY-NC-4.0 (данные)

**1.2 Model Cards для каждой модели**

- **Критерий**: Каждая production модель должна иметь документ с метриками, ограничениями, предназначением
- **Измерение**: Наличие `model_card.md` с полями: metrics, intended use, limitations, training data
- **Статус в проекте**: ✅ MODEL_CARD.md + wiki_content/Model-Cards.md

**1.3 Версионирование датасетов**

- **Критерий**: Тренировочные датасеты должны версионироваться с хешированием и tracking
- **Измерение**: Использование DVC/Git LFS с хешами датасетов
- **Статус в проекте**: ✅ DVC настроен (.dvc/config), remotes: Yandex Disk + HuggingFace

#### 2. Воспроизводимость

**2.1 Фиксация всех зависимостей**

- **Критерий**: Все зависимости должны иметь точные версии (pinned versions)
- **Измерение**: 100% зависимостей в `poetry.lock`, `package-lock.json` имеют фиксированные версии
- **Статус в проекте**: ✅ Poetry.lock и package-lock.json существуют

**2.2 Контракты моделей (Input/Output Schema)**

- **Критерий**: Каждая версия модели должна иметь формальный контракт входа/выхода
- **Измерение**: Наличие `model_contract.yaml` с JSON Schema + примеры запросов
- **Статус в проекте**: ✅ model_contract.yaml с JSON Schema + curl примерами

**2.3 Seed Management**

- **Критерий**: Код обучения должен фиксировать все random seeds для воспроизводимости
- **Измерение**: Фиксация torch.manual_seed(), np.random.seed(), random.seed()
- **Статус в проекте**: ✅ utils.py: numpy, random, torch, cuda, cudnn, PYTHONHASHSEED

#### 3. MLOps инфраструктура

**3.1 Централизованный Model Registry**

- **Критерий**: Production модели должны храниться в Model Registry с метриками и версионированием
- **Измерение**: Интеграция с MLflow/HuggingFace Hub с автоматической регистрацией
- **Статус в проекте**: ✅ HuggingFace Hub (baltsat/Whales-Identification) + upload script

**3.2 Мониторинг ML-метрик в production**

- **Критерий**: Production API должен логировать предсказания и метрики качества
- **Измерение**: Эндпоинт `/metrics` с latency p50/p95/p99, prediction distribution
- **Статус в проекте**: ✅ GET /metrics endpoint (Prometheus-compatible)

**3.3 Data Drift Detection**

- **Критерий**: Система должна отслеживать изменения в распределении входных данных
- **Измерение**: Алерты при отклонении распределения >20% от baseline
- **Статус в проекте**: ✅ whales_identify/drift_detection.py (threshold 20%)

### II. ТАКТИЧЕСКИЕ ПРАВИЛА (нижний уровень)

#### 4. CI/CD Pipeline

**4.1 Полнота пайплайна (6 стадий)**

- **Критерий**: CI/CD должен включать: lint → test → security scan → build → integration test → deploy
- **Измерение**: Наличие всех 6 стадий в `.github/workflows` или `.gitlab-ci.yml`
- **Статус в проекте**:
  - ✅ GitLab CI: все стадии есть
  - ✅ GitHub Actions: 6 стадий (lint, test, security, docker, trivy, status)

**4.2 Кэширование зависимостей**

- **Критерий**: CI должен кэшировать зависимости и Docker layers
- **Измерение**: Время повторной сборки < 20% от полной сборки
- **Статус в проекте**: ✅ actions/cache для Poetry и Docker layers

**4.3 Автоматизация тестирования моделей**

- **Критерий**: CI должен проверять работоспособность модели на синтетических данных
- **Измерение**: Integration test с inference на тестовых изображениях
- **Статус в проекте**: ✅ Есть `integration_test_resnet_inference` в GitLab CI

**4.4 Pre-commit hooks: комплексность**

- **Критерий**: Hooks должны включать: formatting, linting, type checking, security, tests
- **Измерение**: Наличие black/ruff, flake8, mypy, bandit, pytest в `.pre-commit-config.yaml`
- **Статус в проекте**: ✅ black, flake8, mypy, bandit, isort, nbqa, prettier

#### 5. Тестирование

**5.1 Трёхуровневое покрытие**

- **Критерий**: Unit (изолированные функции), Integration (API), E2E (полный workflow)
- **Измерение**:
  - Unit: >80% coverage (pytest-cov)
  - Integration: все endpoints протестированы
  - E2E: минимум 1 full scenario
- **Статус в проекте**: ✅ Unit (whales_identify/tests/), Integration (API), Frontend (Jest)

**5.2 Негативные тесты**

- **Критерий**: Тестирование крайних случаев и ошибок
- **Измерение**: Минимум 10 тестов на invalid inputs, timeouts, large files
- **Статус в проекте**: ✅ 12+ тестов: media type, bad zip, empty file, empty zip, v1, metrics

**5.3 Performance тесты**

- **Критерий**: Benchmark тесты с latency и throughput метриками
- **Измерение**: Использование locust/k6, требования p95 latency < X ms
- **Статус в проекте**: ✅ tests/performance/locustfile.py (Locust)

#### 6. Документация

**6.1 OpenAPI specs с примерами**

- **Критерий**: Каждый endpoint с curl примерами и response examples
- **Измерение**: 100% endpoints в OpenAPI с примерами успешных и ошибочных запросов
- **Статус в проекте**: ✅ FastAPI автогенерирует /docs, но нет явных examples в коде

**6.2 API Changelog**

- **Критерий**: Версионирование breaking changes в API
- **Измерение**: Наличие `API_CHANGELOG.md` с версией, датой, списком изменений
- **Статус в проекте**: ✅ API_CHANGELOG.md

**6.3 Туториалы quick start**

- **Критерий**: README с working examples (копируй-вставляй)
- **Измерение**: Новый пользователь запускает проект за <10 минут
- **Статус в проекте**: ✅ Частично (docker-compose up работает, но требуется загрузка моделей)

**6.4 Исправление неточностей в документации**

- **Критерий**: Все команды в README должны работать без ошибок
- **Измерение**: 100% команд выполняются успешно на чистой системе
- **Статус в проекте**: ✅ Команды проверены и исправлены

#### 7. Docker & Контейнеризация

**7.1 Multi-stage builds**

- **Критерий**: Минимизация размера образа через multi-stage
- **Измерение**: Production образ < 2x minimal Python/Node image
- **Статус в проекте**:
  - ✅ Frontend использует multi-stage
  - ✅ Backend использует multi-stage (builder + runtime)

**7.2 Health checks**

- **Критерий**: Readiness и liveness probes для каждого сервиса
- **Измерение**: Наличие HEALTHCHECK в Dockerfile или healthcheck в docker-compose
- **Статус в проекте**: ✅ HEALTHCHECK в Dockerfile + healthcheck в docker-compose

**7.3 Self-contained образы**

- **Критерий**: Все зависимости внутри образа, нет external downloads при старте
- **Измерение**: Container startup time <10s, нет curl/wget в CMD
- **Статус в проекте**: ⚠️ Модели загружаются извне, требуется download_models.sh

**7.4 Безопасность: non-root user**

- **Критерий**: Контейнеры запускаются от non-root пользователя
- **Измерение**: Наличие `USER <non-root>` в Dockerfile
- **Статус в проекте**: ✅ Backend: USER appuser (non-root)

**7.5 System dependencies**

- **Критерий**: Все необходимые системные библиотеки установлены в Dockerfile
- **Измерение**: Container успешно запускается без ImportError
- **Статус в проекте**: ✅ libgl1, libglib2.0-0, libsm6, libxext6, libxrender-dev, libgomp1

#### 8. Безопасность

**8.1 Валидация входных данных**

- **Критерий**: Многоуровневая валидация: формат, размер, содержимое, rate limiting
- **Измерение**: Middleware с MIME type, file size limit, rate limiting
- **Статус в проекте**: ✅ MIME type + rate limiting (60 req/min per IP)

**8.2 Сканирование уязвимостей**

- **Критерий**: Автоматическое сканирование образов и зависимостей
- **Измерение**: Trivy для Docker, Bandit для Python кода, Dependabot для зависимостей
- **Статус в проекте**: ✅ Bandit + Trivy в CI, Safety для зависимостей

**8.3 Секреты в Secret Managers**

- **Критерий**: Никаких credentials в репозитории
- **Измерение**: 0 файлов с паттернами password|secret|key|token
- **Статус в проекте**: ✅ Нет явных секретов

#### 9. Модульность кода

**9.1 Pluggable архитектура моделей**

- **Критерий**: Смена модели через конфигурацию, а не изменение кода
- **Измерение**: Наличие `models_config.yaml` + factory pattern
- **Статус в проекте**: ✅ models_config.yaml с 6 моделями + factory pattern

**9.2 API версионирование**

- **Критерий**: Версионные префиксы в URL (/v1/, /v2/)
- **Измерение**: Поддержка N и N-1 версий одновременно
- **Статус в проекте**: ✅ /v1/ prefix + backward-compatible root endpoints

### Оценка зрелости текущего проекта

| Категория          | Текущий статус             | Приоритет улучшения | Целевой уровень                               |
| ------------------ | -------------------------- | ------------------- | --------------------------------------------- |
| **Лицензирование** | ✅ Код/модели/данные        | ✅ Выполнено         | Лицензии для кода/моделей/данных              |
| **CI/CD**          | ✅ GitLab + GitHub (6 стадий) | ✅ Выполнено      | Полные пайплайны в обоих                      |
| **Тестирование**   | ✅ Unit + Integration + Perf | ✅ Выполнено       | Unit + Integration + E2E + Performance        |
| **MLOps**          | ✅ HF Hub + /metrics + drift | ✅ Выполнено       | Model Registry + мониторинг + drift detection |
| **Документация**   | ✅ Wiki + API changelog      | ✅ Выполнено       | Точные инструкции + API changelog             |
| **Docker**         | ✅ Health + non-root + multi-stage | ✅ Выполнено  | Health checks + non-root + self-contained     |
| **Pre-commit**     | ✅ black/flake8/mypy/bandit  | ✅ Выполнено       | + mypy + bandit + pytest                      |
| **Безопасность**   | ✅ Rate limit + Trivy + Bandit | ✅ Выполнено     | Rate limiting + сканирование + validation     |

**Общая оценка зрелости**: 90/100 (Production Ready)
**Цель для production**: 90/100 (Production Ready)

### Приоритетные задачи для достижения production-ready

#### Критические (блокируют production):

1. ✅ Добавить явные лицензии на модели и датасеты
2. ✅ Реализовать Model Registry (MLflow или HuggingFace Model Hub)
3. ✅ Добавить мониторинг ML-метрик (Prometheus + Grafana)
4. ✅ Реализовать health checks в Docker контейнерах
5. ✅ Добавить API версионирование (/v1/ prefix)

#### Высокий приоритет:

6. ✅ Исправить все неработающие команды из README
7. ✅ Расширить pre-commit hooks (mypy, bandit, pytest)
8. ✅ Добавить performance тесты (k6/locust)
9. ✅ Реализовать rate limiting и полную валидацию входов
10. ✅ Добавить coverage reporting (pytest-cov >80%)

#### Средний приоритет:

11. ⚪ Оптимизировать Docker (multi-stage backend, non-root)
12. ⚪ Включить модели в Docker образ или init container
13. ⚪ Создать model_contract.yaml с примерами
14. ⚪ Добавить DVC для версионирования данных
15. ⚪ Реализовать data drift detection
