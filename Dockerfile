FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and poetry.lock into the container
COPY pyproject.toml poetry.lock /app/

# Install dependencies with Poetry
RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-ansi

# Copy the rest of the application
COPY . /app/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the application
CMD ["python", "train.py"]