install:
    poetry install

lint:
    poetry run flake8 .

test:
    poetry run pytest --maxfail=1 --disable-warnings -q

lint-and-test: lint test

run:
    poetry run python train.py

lint:
    poetry run pylint .