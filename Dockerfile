FROM python:3.10.15-slim

WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-ansi

COPY . /app/

EXPOSE 8080
