FROM python:3.10.15-slim

WORKDIR /app

# Установка Poetry через pip
RUN pip install poetry

COPY pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-ansi

COPY . /app/

EXPOSE 8080
