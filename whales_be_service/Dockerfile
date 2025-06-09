# -------- export-deps stage --------
FROM python:3.11-slim AS export
WORKDIR /app

# ставим Poetry и экспортируем prod-зависимости
RUN pip install --no-cache-dir poetry==1.8.2
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry export --without-hashes -f requirements.txt -o requirements.txt

# -------- final app stage --------
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# deps + uvicorn
COPY --from=export /app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt uvicorn fastapi python-multipart

# копируем исходники
COPY src ./src

# чтобы Python нашёл whales_be_service в src/
ENV PYTHONPATH=/app/src

EXPOSE 8000

# запускаем через python -m uvicorn
CMD ["python", "-m", "uvicorn", "whales_be_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
