FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY main.py .

# Экспорт порта
EXPOSE 8080

# Запуск приложения
CMD ["python", "main.py"] 