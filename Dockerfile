FROM python:3.10.15-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-ansi

COPY . /app/

EXPOSE 8080 8051

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./research/demo-ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]