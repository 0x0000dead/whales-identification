# Установить зависимости через Poetry
install:
    poetry install

# Запуск линтинга (flake8 или pylint)
lint:
    poetry run flake8 .

# Запуск тестов через pytest
test:
    poetry run pytest --maxfail=1 --disable-warnings -q

# Запуск линтинга и тестов вместе
lint-and-test: lint test

# Запуск приложения
run:
    poetry run python train.py
