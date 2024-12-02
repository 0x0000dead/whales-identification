# Unit Tests Documentation

This document provides an overview of the unit tests implemented for the **whales-identification** project.

## Introduction

Unit tests are essential for ensuring the reliability and correctness of the codebase. They help detect bugs early and make the code more maintainable. This project includes unit tests for critical components, focusing on the most important functionalities.

## Test Structure

The tests are located in the `tests` directory and cover the following modules:

- `utils.py`
- `filter_processor.py`
- `model.py`

### Test Files

- `test_utils.py`: Tests utility functions.
- `test_filter_processor.py`: Tests the data filtering process.
- `test_model.py`: Tests core model components.

## Test Overview

### 1. Utility Functions Tests (`test_utils.py`)

**Tested Functionality:**

- `set_seed` Function

Ensures that setting the seed results in reproducible random numbers across different libraries.

**Test Logic:**

1. Set a specific seed using `set_seed(42)`.
2. Generate random numbers from `torch`, `numpy`, and the standard `random` module.
3. Reset the seed to the same value and generate the numbers again.
4. Assert that the numbers from both sets are equal, confirming reproducibility.

### 2. Filter Processor Tests (`test_filter_processor.py`)

**Tested Functionality:**

- `FilterProcessor.filter_data` Method

Verifies that the method correctly filters out entries with invalid labels from the dataset.

**Test Logic:**

1. Create a dummy dataset with both valid and invalid labels.
2. Instantiate `FilterProcessor` with a list of valid labels.
3. Apply the `filter_data` method to the dataset.
4. Assert that only entries with valid labels remain.

### 3. Model Components Tests (`test_model.py`)

**Tested Components:**

- `GeM` Class

    Tests the Generalized Mean Pooling layer's output shape.

- `ArcMarginProduct` Class

    Verifies the output shape and functionality of the ArcFace loss layer.

- `HappyWhaleModel` Class

    Checks the forward pass and output dimensions.

**Test Logic:**

- **GeM Test:**
    1. Create a random tensor simulating feature maps.
    2. Pass it through the `GeM` layer.
    3. Assert that the output has the expected dimensions.

- **ArcMarginProduct Test:**
    1. Generate random embeddings and labels.
    2. Pass them through the `ArcMarginProduct` layer.
    3. Assert that the output tensor has the correct shape.

- **HappyWhaleModel Test:**
    1. Instantiate the model with sample parameters.
    2. Create synthetic image data and labels.
    3. Perform a forward pass.
    4. Assert that the output logits have the correct shape.

## Running the Tests

### Prerequisites

Ensure you have all project dependencies installed. Use **poetry** for dependency management.

```bash
# Install dependencies using poetry
poetry install
```

### Executing Tests

Run the following command from the project root directory:

```bash
poetry run python -m unittest discover tests
```

This command discovers and runs all tests in the `tests` directory.

## Continuous Integration

The project utilizes GitHub Actions for automated testing on every push and pull request to the `main` branch.

**Workflow File:** `python-app.yml`

**Workflow Steps:**

1. **Checkout Code:**

     ```yaml
     - uses: actions/checkout@v4
     ```

2. **Set Up Python Environment:**

     ```yaml
     - name: Set up Python
         uses: actions/setup-python@v4
         with:
             python-version: '3.10'
     ```

3. **Install Dependencies:**

     ```yaml
     - name: Install dependencies
         run: |
             pip install poetry
             poetry install
     ```

4. **Run Tests:**

     ```yaml
     - name: Run tests
         run: |
             poetry run python -m unittest discover tests
     ```

## Conclusion

By focusing on these key functionalities, the unit tests help ensure that critical parts of the codebase work correctly. This enhances the project's reliability and facilitates future development and maintenance.