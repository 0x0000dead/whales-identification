# -------- export-deps stage --------
FROM python:3.11-slim AS export
WORKDIR /app

# Установка системных зависимостей для OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ставим Poetry и экспортируем prod-зависимости
RUN pip install --no-cache-dir poetry==1.8.2
COPY pyproject.toml poetry.lock ./
# Используем --no-root чтобы не устанавливать проект целиком
RUN poetry config virtualenvs.create false \
 && poetry export --without-hashes -f requirements.txt -o requirements.txt

# -------- final app stage --------
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей для OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# deps + uvicorn + дополнительные ML зависимости
COPY --from=export /app/requirements.txt ./
RUN pip install --no-cache-dir \
    -r requirements.txt \
    uvicorn[standard]==0.24.0 \
    fastapi==0.104.1 \
    python-multipart==0.0.6 \
    opencv-python-headless==4.10.0.84 \
    torch==2.4.1 \
    torchvision==0.19.1 \
    timm==1.0.9 \
    albumentations==1.4.18 \
    pillow==10.4.0 \
    numpy==1.26.0 \
    psutil==5.9.8

# копируем исходники
COPY src ./src

# Создание директории для моделей
RUN mkdir -p /app/models

# чтобы Python нашёл whales_be_service в src/
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# запускаем через python -m uvicorn
CMD ["python", "-m", "uvicorn", "whales_be_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
