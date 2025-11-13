# Contributing Guide

Welcome to the Whales Identification project! This guide will help you get started with contributing.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Review Guidelines](#code-review-guidelines)
- [Common Tasks](#common-tasks)

---

## Getting Started

### Prerequisites

- Python 3.11.6
- Node.js ≥16
- Docker & Docker Compose
- Git
- Poetry
- Text editor (VS Code, PyCharm, etc.)

### Initial Setup

#### 1. Fork and Clone

```bash
# Fork on GitHub first, then clone
git clone https://github.com/YOUR_USERNAME/whales-identification.git
cd whales-identification

# Add upstream remote
git remote add upstream https://github.com/0x0000dead/whales-identification.git
```

#### 2. Backend Setup

```bash
cd whales_be_service

# Install Poetry (if not installed)
pip install poetry

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Verify installation
poetry run pytest --version
```

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Verify
npm run dev
```

#### 4. Download Models

```bash
# From project root
pip install huggingface_hub
./scripts/download_models.sh
```

#### 5. Verify Setup

```bash
# Run tests
cd whales_be_service
poetry run pytest

# Start services
cd ..
docker compose up --build
```

---

## Development Workflow

### Git Workflow

We use **Feature Branch Workflow** with the following conventions:

#### Branch Naming

```
<type>/<short-description>

Types:
- feature/  - New features
- fix/      - Bug fixes
- docs/     - Documentation changes
- refactor/ - Code refactoring
- test/     - Test additions/fixes
- chore/    - Maintenance tasks

Examples:
- feature/add-orca-detection
- fix/login-bug
- docs/update-api-reference
- refactor/optimize-inference
```

#### Commit Messages

Follow **Conventional Commits:**

```
<type>(<scope>): <subject>

<body>

<footer>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Code style (formatting, no logic change)
- refactor: Code refactoring
- test: Tests
- chore: Maintenance

Examples:
feat(api): add batch prediction endpoint

Implement ZIP upload and batch processing for
multiple whale images. Supports up to 100 images
per request.

Closes #42

---

fix(inference): resolve memory leak in model loading

Model was not being properly released after inference,
causing memory usage to grow over time. Now properly
using context managers.

Fixes #67

---

docs(wiki): add API reference documentation

Complete documentation for all API endpoints with
curl examples and response schemas.
```

#### Daily Workflow

```bash
# 1. Start your day - sync with upstream
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/my-new-feature

# 3. Make changes and commit frequently
git add .
git commit -m "feat(scope): description"

# 4. Keep branch updated
git fetch upstream
git rebase upstream/main

# 5. Push to your fork
git push origin feature/my-new-feature

# 6. Open Pull Request on GitHub

# 7. After PR is merged, cleanup
git checkout main
git pull upstream main
git branch -d feature/my-new-feature
```

---

## Code Style

### Python

#### PEP 8 with Black Formatting

```python
# Good - follows black (line length 88)
def predict_whale_species(
    image_path: str,
    model: HappyWhaleModel,
    config: dict,
) -> Detection:
    """
    Predict whale species from image.

    Args:
        image_path: Path to whale image
        model: Trained model instance
        config: Configuration dictionary

    Returns:
        Detection object with prediction results
    """
    image = load_image(image_path)
    tensor = preprocess(image)

    with torch.no_grad():
        embeddings = model(tensor)
        prediction = model.classify(embeddings)

    return Detection(**prediction)


# Bad - inconsistent formatting
def predict_whale_species(image_path,model,config):
    image=load_image(image_path);tensor=preprocess(image)
    with torch.no_grad():embeddings=model(tensor);prediction=model.classify(embeddings)
    return Detection(**prediction)
```

#### Type Hints

```python
# Good - explicit types
from typing import Optional, List, Dict

def process_batch(
    images: List[np.ndarray],
    batch_size: int = 32
) -> List[Detection]:
    results: List[Detection] = []
    for batch in create_batches(images, batch_size):
        predictions = model.predict(batch)
        results.extend(predictions)
    return results


# Bad - no type hints
def process_batch(images, batch_size=32):
    results = []
    for batch in create_batches(images, batch_size):
        predictions = model.predict(batch)
        results.extend(predictions)
    return results
```

#### Docstrings

```python
# Good - Google style docstrings
def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predictions: Model predictions (N, C)
        targets: Ground truth labels (N,)

    Returns:
        Dictionary with metrics:
            - precision: Precision@1
            - recall: Recall
            - f1: F1-score

    Raises:
        ValueError: If predictions and targets have different lengths

    Examples:
        >>> predictions = torch.randn(10, 5)
        >>> targets = torch.randint(0, 5, (10,))
        >>> metrics = calculate_metrics(predictions, targets)
        >>> print(metrics['precision'])
        0.85
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    precision = compute_precision(predictions, targets)
    recall = compute_recall(predictions, targets)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
```

### TypeScript/React

#### ESLint + Prettier

```typescript
// Good - proper formatting and types
interface WhaleDetection {
  imageInd: string;
  classAnimal: string;
  idAnimal: string;
  probability: number;
  bbox: [number, number, number, number];
  mask: string | null;
}

const ResultDisplay: React.FC<{ detection: WhaleDetection }> = ({
  detection,
}) => {
  const confidencePercent = (detection.probability * 100).toFixed(1);

  return (
    <div className="result-container">
      <h2>{detection.idAnimal}</h2>
      <p>Confidence: {confidencePercent}%</p>
      {detection.mask && (
        <img
          src={`data:image/png;base64,${detection.mask}`}
          alt="Whale mask"
        />
      )}
    </div>
  );
};

// Bad - no types, inconsistent formatting
const ResultDisplay = ({ detection }) => {
  const confidencePercent = detection.probability * 100;
  return (
    <div>
      <h2>{detection.idAnimal}</h2>
      <p>Confidence: {confidencePercent}%</p>
      {detection.mask && (
        <img src={`data:image/png;base64,${detection.mask}`} />
      )}
    </div>
  );
};
```

---

## Pre-commit Hooks

### Installation

```bash
cd whales_be_service
poetry run pre-commit install
```

### Hooks Configuration

We use 20 pre-commit hooks:

| Category | Hooks | Auto-fix |
|----------|-------|----------|
| **Formatting** | black, isort, prettier | ✅ Yes |
| **Linting** | flake8 | ❌ No |
| **Type Checking** | mypy | ❌ No |
| **Security** | bandit | ❌ No |
| **Jupyter** | nbstripout, nbqa-* | ✅ Partial |
| **Basic** | trailing-whitespace, end-of-file-fixer, etc. | ✅ Most |

**See [PRE_COMMIT_GUIDE.md](https://github.com/0x0000dead/whales-identification/blob/main/docs/PRE_COMMIT_GUIDE.md) for full documentation.**

### Running Manually

```bash
# Run on staged files (automatic on commit)
poetry run pre-commit run

# Run on all files
poetry run pre-commit run --all-files

# Run specific hook
poetry run pre-commit run black --all-files

# Skip hooks (NOT RECOMMENDED)
git commit --no-verify -m "Emergency fix"
```

### Common Hook Failures

#### Black formatting

```bash
# Failure: Code not formatted
# Fix: Run black
poetry run black .
git add .
git commit -m "style: format code with black"
```

#### Flake8 linting

```bash
# Failure: F401: Module imported but unused
# Fix: Remove unused import
-import pandas as pd  # Not used
+# pandas not needed

# Failure: E501: Line too long (>88 characters)
# Fix: Break line
-very_long_string = "This is an extremely long string that exceeds the maximum line length"
+very_long_string = (
+    "This is an extremely long string "
+    "that exceeds the maximum line length"
+)
```

#### Mypy type checking

```bash
# Failure: Missing type hints
# Fix: Add type hints
-def process(data):
+def process(data: List[str]) -> Dict[str, int]:
    return {"count": len(data)}
```

---

## Testing

### Running Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=term

# Fast tests only (skip slow integration tests)
poetry run pytest -m "not slow"

# Specific test file
poetry run pytest tests/api/test_post_endpoints.py -v
```

### Writing Tests

```python
# tests/unit/test_new_feature.py

import pytest
from whales_be_service.new_feature import new_function

def test_new_function_success():
    """Test new_function with valid input"""
    result = new_function(input_data="test")
    assert result == expected_output

def test_new_function_invalid_input():
    """Test new_function handles invalid input"""
    with pytest.raises(ValueError):
        new_function(input_data=None)

@pytest.mark.slow
def test_new_function_integration():
    """Test new_function with real model (slow)"""
    model = load_full_model()
    result = new_function(model=model, data=test_data)
    assert result is not None
```

**See [Testing Guide](Testing) for comprehensive testing documentation.**

---

## Pull Request Process

### Before Opening PR

**Checklist:**
- [ ] Code follows style guide (black, flake8, mypy pass)
- [ ] All tests pass (`poetry run pytest`)
- [ ] New tests added for new features
- [ ] Documentation updated (docstrings, wiki)
- [ ] Pre-commit hooks pass
- [ ] Branch is up-to-date with main
- [ ] Commit messages follow conventions

### Opening PR

1. **Push to your fork:**
   ```bash
   git push origin feature/my-feature
   ```

2. **Open PR on GitHub:**
   - Navigate to https://github.com/0x0000dead/whales-identification
   - Click "Pull requests" → "New pull request"
   - Select your branch

3. **Fill PR template:**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Unit tests added
   - [ ] Integration tests added
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guide
   - [ ] Tests pass
   - [ ] Documentation updated

   ## Related Issues
   Closes #42
   ```

### PR Review Process

1. **Automated checks run:**
   - Linting (black, flake8, isort, mypy)
   - Security (bandit, safety)
   - Tests (pytest with coverage)
   - Docker build

2. **Code review:**
   - At least 1 approval required
   - Reviewers check:
     - Code quality
     - Test coverage
     - Documentation
     - Security

3. **Address feedback:**
   ```bash
   # Make changes
   git add .
   git commit -m "fix: address review feedback"
   git push origin feature/my-feature
   ```

4. **Merge:**
   - After approval, maintainer merges PR
   - Delete feature branch

---

## Code Review Guidelines

### As a Reviewer

**What to check:**
- ✅ Code correctness and logic
- ✅ Test coverage (>80% for new code)
- ✅ Documentation and comments
- ✅ Security issues (SQL injection, XSS, etc.)
- ✅ Performance implications
- ✅ Consistency with existing code

**How to provide feedback:**
```markdown
# Good - constructive, specific
Consider using a list comprehension here for better readability:
``python
# Instead of
results = []
for item in items:
    results.append(process(item))

# Use
results = [process(item) for item in items]
``

# Bad - vague, unhelpful
This code is bad.
```

### As an Author

**Responding to feedback:**
```markdown
# Good - acknowledge, explain, implement
Thanks for the suggestion! You're right that a list comprehension
is cleaner here. I've updated the code in commit abc123.

# Bad - defensive
My code is fine. This is just your opinion.
```

---

## Common Tasks

### Adding a New API Endpoint

1. **Define endpoint in routers.py:**
```python
@router.post("/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest):
    # Implementation
    return NewResponse(...)
```

2. **Add Pydantic models:**
```python
# response_models.py
class NewRequest(BaseModel):
    field: str

class NewResponse(BaseModel):
    result: str
```

3. **Write tests:**
```python
# tests/api/test_new_endpoint.py
def test_new_endpoint_success(client):
    response = client.post("/new-endpoint", json={"field": "value"})
    assert response.status_code == 200
```

4. **Update documentation:**
   - API Reference wiki page
   - OpenAPI schema (auto-generated by FastAPI)

### Adding a New Model

1. **Create model class:**
```python
# whales_identify/models/new_model.py
class NewModel(nn.Module):
    def __init__(self, ...):
        ...
```

2. **Add training script:**
```python
# whales_identify/train_new_model.py
def train_new_model():
    ...
```

3. **Create model card:**
   - Add to Model-Cards wiki page
   - Include metrics, intended use, limitations

4. **Add integration:**
```python
# whales_be_service/whale_infer.py
if model_type == "new_model":
    self.model = NewModel.load(model_path)
```

### Updating Dependencies

```bash
# Backend
cd whales_be_service
poetry add package_name
poetry lock
git add pyproject.toml poetry.lock

# Frontend
cd frontend
npm install package_name
git add package.json package-lock.json

# Commit
git commit -m "chore: add package_name dependency"
```

---

## Community

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Questions, ideas

### Getting Help

1. **Check documentation first:**
   - Wiki pages
   - README
   - Code comments

2. **Search existing issues:**
   - Someone may have had the same problem

3. **Open a new issue:**
   - Provide context
   - Include error messages
   - Share minimal reproducible example

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project:
- **Code:** MIT License
- **Models:** Apache 2.0 (with restrictions)
- **Data:** CC-BY-NC-4.0

See [LICENSE](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE), [LICENSE_MODELS.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_MODELS.md), and [LICENSE_DATA.md](https://github.com/0x0000dead/whales-identification/blob/main/LICENSE_DATA.md).

---

## Thank You!

Thank you for contributing to Whales Identification! Your contributions help protect marine mammals. 🐋

---

**Related Pages:**
- [Installation](Installation) - Setup development environment
- [Testing](Testing) - Testing guidelines
- [Architecture](Architecture) - System design
- [API Reference](API-Reference) - API documentation
