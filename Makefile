install:
    poetry install

lint:
    poetry run black --check .
    poetry run flake8 .

lint-fix:
    poetry run black .

pylint:
    poetry run pylint .

test:
    poetry run pytest --maxfail=1 --disable-warnings -q

lint-and-test: lint test

run:
    poetry run python train.py