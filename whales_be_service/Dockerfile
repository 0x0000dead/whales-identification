FROM python:3.11.6-slim

# 1) Системные зависимости для сборки
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libssl-dev \
      libffi-dev \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Устанавливаем Poetry
RUN pip install --no-cache-dir poetry==1.8.5

# 3) Копируем pyproject + lock и сразу ставим prod-зависимости
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-dev --no-interaction --no-ansi

# 4) Копируем исходники
COPY src ./src

# 5) Переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

# 6) Запуск через модуль uvicorn
CMD ["python", "-m", "uvicorn", "whales_be_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
