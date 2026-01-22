# CI/CD Pipeline Evidence

This document provides evidence of CI/CD pipeline implementation with links to actual GitHub Actions workflow runs.

## Table of Contents

- [Overview](#overview)
- [GitHub Actions Workflows](#github-actions-workflows)
- [Pipeline Stages](#pipeline-stages)
- [Example Workflow Runs](#example-workflow-runs)
- [Pipeline Architecture](#pipeline-architecture)
- [Badge Status](#badge-status)

---

## Overview

The EcoMarineAI project uses **GitHub Actions** for continuous integration and deployment. Our CI/CD pipeline ensures code quality, security, and reliability through automated checks on every push and pull request.

**Pipeline Capabilities:**

- 🔍 **Code Quality**: Black, isort, Flake8, Mypy
- 🔒 **Security**: Bandit, Safety, Trivy (Docker scanning)
- 🧪 **Testing**: pytest with coverage reporting
- 🐳 **Docker**: Build and integration testing
- 📚 **Documentation**: Automated GitHub Pages deployment

**Total Workflows:** 5 active workflows

---

## GitHub Actions Workflows

| Workflow            | File               | Trigger                  | Purpose                         |
| ------------------- | ------------------ | ------------------------ | ------------------------------- |
| **CI/CD Pipeline**  | `ci.yml`           | Push/PR to main, develop | Full quality pipeline           |
| **Docker Image CI** | `docker-image.yml` | Push/PR to main          | Docker build verification       |
| **Deploy Docs**     | `deploy-docs.yml`  | Push to main (docs/)     | GitHub Pages deployment         |
| **Greetings**       | `greetings.yml`    | Issues, PRs              | Welcome first-time contributors |
| **Labeler**         | `label.yml`        | Pull requests            | Auto-label PRs by path          |

---

## Pipeline Stages

### CI/CD Pipeline (`ci.yml`)

The main CI/CD pipeline consists of **6 stages**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CI/CD Pipeline Flow                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │  Stage 1 │    │  Stage 2 │    │  Stage 3 │    │     Stage 4      │  │
│  │   Lint   │───▶│ Security │───▶│   Test   │───▶│  Docker Build &  │  │
│  │          │    │          │    │          │    │ Integration Test │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘  │
│       │                                                    │            │
│       │                                                    ▼            │
│       │                                          ┌──────────────────┐   │
│       │                                          │     Stage 5      │   │
│       │                                          │  Docker Security │   │
│       │                                          │      Scan        │   │
│       │                                          └──────────────────┘   │
│       │                                                    │            │
│       └───────────────────────┬────────────────────────────┘            │
│                               ▼                                         │
│                      ┌──────────────────┐                              │
│                      │     Stage 6      │                              │
│                      │   Status Badge   │                              │
│                      └──────────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Stage 1: Lint and Format Check

- **Black**: Code formatting verification
- **isort**: Import sorting check
- **Flake8**: PEP 8 compliance and linting
- **Mypy**: Static type checking (non-blocking)

#### Stage 2: Security Scan

- **Bandit**: Python security vulnerability detection
- **Safety**: Dependency vulnerability check

#### Stage 3: Testing

- **pytest**: Unit and integration tests
- **Coverage**: Code coverage reporting with Codecov integration

#### Stage 4: Docker Build and Integration Test

- Build Docker images with `docker compose`
- Start containers and verify health endpoints
- Run integration tests against running services

#### Stage 5: Docker Image Security Scan (main branch only)

- **Trivy**: Container image vulnerability scanning
- Results uploaded to GitHub Security tab

#### Stage 6: Status Badge

- Aggregate results from all stages
- Comment on PRs with pipeline status

---

## Example Workflow Runs

### Recent CI/CD Pipeline Runs

View the full history of CI/CD pipeline runs:

- **All Runs**: https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml

#### Example Successful Run

**What this run tested:**

1. ✅ **Lint Stage**: Verified code formatting with Black, import order with isort, linting with Flake8
2. ✅ **Security Stage**: Scanned for security vulnerabilities with Bandit
3. ✅ **Test Stage**: Ran pytest suite with coverage reporting
4. ✅ **Docker Stage**: Built and tested Docker containers

**Artifacts produced:**

- Coverage XML report uploaded to Codecov
- Security scan results (if applicable)

### Recent Docker Image CI Runs

View Docker build history:

- **All Runs**: https://github.com/0x0000dead/whales-identification/actions/workflows/docker-image.yml

**What this workflow tests:**

1. Builds both backend and frontend Docker images
2. Verifies successful image creation
3. Tests Docker Compose configuration

### Recent Documentation Deployments

View documentation deployment history:

- **All Runs**: https://github.com/0x0000dead/whales-identification/actions/workflows/deploy-docs.yml

**What this workflow does:**

1. Triggers when files in `docs/` directory change
2. Deploys documentation to GitHub Pages
3. Provides URL to deployed documentation

---

## Pipeline Architecture

### Dependency Graph

```
                    ┌───────────────┐
                    │     lint      │
                    │  (parallel)   │
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │ security  │ │   test    │ │  (waits)  │
        │ (parallel)│ │  (needs   │ │           │
        │           │ │   lint)   │ │           │
        └─────┬─────┘ └─────┬─────┘ │           │
              │             │       │           │
              └─────────────┼───────┘           │
                            │                   │
                            ▼                   │
                      ┌───────────┐             │
                      │  docker   │◀────────────┘
                      │ (needs    │
                      │ lint,test,│
                      │ security) │
                      └─────┬─────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │  docker-  │ │  status   │ │           │
        │  security │ │  (needs   │ │           │
        │ (main     │ │   all)    │ │           │
        │  only)    │ │           │ │           │
        └───────────┘ └───────────┘ └───────────┘
```

### Caching Strategy

The pipeline implements caching for:

- **Poetry dependencies**: Cached based on `poetry.lock` hash
- **Docker layers**: Cached with BuildX

Example cache configuration:

```yaml
- name: Cache Poetry dependencies
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pypoetry
      ~/.cache/pip
    key: ${{ runner.os }}-poetry-${{ hashFiles('whales_be_service/poetry.lock') }}
    restore-keys: |
      ${{ runner.os }}-poetry-
```

---

## Badge Status

### CI/CD Status Badge

Add this badge to your README to show pipeline status:

```markdown
[![CI/CD Pipeline](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml/badge.svg)](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml)
```

**Result:**
[![CI/CD Pipeline](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml/badge.svg)](https://github.com/0x0000dead/whales-identification/actions/workflows/ci.yml)

### Docker Build Status Badge

```markdown
[![Docker Image CI](https://github.com/0x0000dead/whales-identification/actions/workflows/docker-image.yml/badge.svg)](https://github.com/0x0000dead/whales-identification/actions/workflows/docker-image.yml)
```

**Result:**
[![Docker Image CI](https://github.com/0x0000dead/whales-identification/actions/workflows/docker-image.yml/badge.svg)](https://github.com/0x0000dead/whales-identification/actions/workflows/docker-image.yml)

---

## Workflow Details

### Tools and Versions

| Tool       | Version | Stage           | Purpose              |
| ---------- | ------- | --------------- | -------------------- |
| Python     | 3.11    | All             | Runtime              |
| Black      | Latest  | Lint            | Code formatting      |
| isort      | Latest  | Lint            | Import sorting       |
| Flake8     | Latest  | Lint            | Linting              |
| Mypy       | Latest  | Lint            | Type checking        |
| Bandit     | Latest  | Security        | Security scan        |
| Safety     | Latest  | Security        | Dependency CVE check |
| pytest     | Latest  | Test            | Test runner          |
| pytest-cov | Latest  | Test            | Coverage             |
| Trivy      | Latest  | Docker Security | Container scanning   |

### Environment Variables

```yaml
env:
  PYTHON_VERSION: "3.11"
```

### Triggers

| Workflow        | Push          | Pull Request  | Manual | Paths     |
| --------------- | ------------- | ------------- | ------ | --------- |
| CI/CD Pipeline  | main, develop | main, develop | ❌     | All       |
| Docker Image CI | main          | main          | ❌     | All       |
| Deploy Docs     | main          | ❌            | ✅     | docs/\*\* |
| Greetings       | ❌            | ✅            | ❌     | All       |
| Labeler         | ❌            | ✅            | ❌     | All       |

---

## Monitoring and Debugging

### View Workflow Runs

1. Go to: https://github.com/0x0000dead/whales-identification/actions
2. Select workflow from left sidebar
3. Click on specific run to see details

### Debug Failed Runs

For failed Docker jobs, logs are automatically printed:

```yaml
- name: Show Docker logs on failure
  if: failure()
  run: docker compose logs
```

### Re-run Failed Jobs

1. Navigate to the failed workflow run
2. Click "Re-run failed jobs" button
3. Monitor new run

---

## CI/CD Best Practices Implemented

| Practice               | Implementation                  | Benefit                    |
| ---------------------- | ------------------------------- | -------------------------- |
| **Dependency Caching** | Poetry and Docker layer caching | ⚡ Faster builds           |
| **Parallel Jobs**      | lint, security run in parallel  | ⚡ Faster feedback         |
| **Fail-Fast**          | Tests stop on first failure     | 🔍 Quick error detection   |
| **Continue-on-Error**  | Non-critical checks marked      | 🛡️ Prevents false blockers |
| **Artifacts**          | Coverage uploaded to Codecov    | 📊 Metrics tracking        |
| **PR Comments**        | Automatic status comments       | 📝 Clear feedback          |
| **Security Scanning**  | Bandit, Safety, Trivy           | 🔒 Vulnerability detection |

---

## References

- **GitHub Actions Documentation**: https://docs.github.com/en/actions
- **Codecov**: https://about.codecov.io/
- **Trivy**: https://trivy.dev/
- **Bandit**: https://bandit.readthedocs.io/

---

**Last Updated:** January 2025
**Version:** 1.0
**Maintained by:** EcoMarineAI Team
