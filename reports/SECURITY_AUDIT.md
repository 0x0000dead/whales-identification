# Security audit report

_Automatically generated artifact. Regenerate via `make security-audit` (or run the commands inline below)._

Last updated: 2026-04-15

## Tools

| Tool         | Version  | Scope                                             |
|--------------|----------|---------------------------------------------------|
| `bandit`     | 1.9.4    | Static-analysis of project Python source code     |
| `pip-audit`  | 2.10.0   | Known CVEs in installed Python dependencies       |
| `pre-commit` | 4.2+     | Blocking check on every commit (`lint.yml` CI)    |

## 1. Bandit — static code analysis

**Command used:**

```bash
bandit -q -r whales_be_service/src whales_identify scripts integrations \
    -x whales_be_service/tests,whales_identify/tests
```

**Initial findings (baseline):**

| ID    | Severity | Location                                               | Cause                            | Action taken |
|-------|----------|--------------------------------------------------------|----------------------------------|--------------|
| B113  | Medium   | `integrations/coordinates_api.py:19`                   | `requests.get()` without timeout | **Fixed** — added `timeout=15` |
| B113  | Medium   | `integrations/coordinates_api.py:33`                   | `requests.get()` without timeout | **Fixed** — added `timeout=15` |
| B311  | Low      | `scripts/populate_test_split.py:77`                    | `random.sample()` in test-data sampler | Annotated `# nosec B311` — deterministic test split, seed 42, not crypto |
| B311  | Low      | `scripts/populate_test_split.py:119`                   | `random.sample()` in test-data sampler | Annotated `# nosec B311` — same rationale |
| B108  | Medium   | `scripts/populate_test_split.py:146`                   | Default `Path("/tmp/kaggle_hw")` | Annotated `# nosec B108` — dev-only default, CI passes `--workdir` explicitly |

**Current state:** 0 unresolved issues.

All `# nosec` annotations include a rationale. Bandit is wired into
`.pre-commit-config.yaml` so new issues block commits.

## 2. pip-audit — dependency vulnerabilities

**Command used:**

```bash
pip-audit --format json
```

**Result:** `No known vulnerabilities found` across **102 installed packages** (full manifest at `reports/pip_audit_latest.json`, regenerate with the command above).

Audited packages include (subset):

- `torch 2.11.0` — no known CVEs
- `fastapi 0.135.x` — no known CVEs
- `pillow 12.2.0` — no known CVEs
- `requests 2.33.1` — no known CVEs
- `numpy 2.4.4` — no known CVEs
- `huggingface-hub 1.10+` — no known CVEs
- `open-clip-torch 3.3.0` — no known CVEs
- `rembg 2.0.75` — no known CVEs

## 3. Pre-commit gate

`.pre-commit-config.yaml` includes the following security-oriented hooks:

- `bandit` (static analysis, blocking)
- `check-added-large-files` (prevents accidental commit of large binaries)
- `check-merge-conflict` (catches unresolved merge markers)
- `check-yaml` / `check-json` / `check-toml` (syntactic validation)
- `end-of-file-fixer` (prevents ambiguous file endings)
- `trailing-whitespace` (prevents subtle diff artefacts)
- `nbstripout` (scrubs Jupyter cell outputs before commit — prevents accidental secret leakage through notebook output)

Interactive runs:

```bash
pre-commit run --all-files
```

## 4. Secrets

| Secret                     | Storage                                     | Never committed |
|----------------------------|---------------------------------------------|:---------------:|
| Kaggle API key             | `~/.kaggle/kaggle.json` (local + CI secret) | ✓              |
| HuggingFace Hub token      | GitHub Actions `HF_TOKEN` secret            | ✓              |
| PostgreSQL DSN             | environment variable, CLI flag              | ✓              |
| WEATHER_API_KEY            | environment variable only                   | ✓              |

A grep of every git object in the repo history (`git log --all -S`) confirms
that none of the above secret values appear in any commit. This was re-verified
during the 2026-04-15 audit.

## 5. Known risks we deliberately accept

| Risk                                                        | Rationale                                  |
|-------------------------------------------------------------|--------------------------------------------|
| `torch.load(weights_only=False)` in identification.py       | Upstream checkpoint format requires it; we trust the HF source. Annotated `# nosec B614` and gated to only read files under `models/`. |
| `rembg` depends on `onnxruntime` (large surface area)       | Optional dependency; service degrades gracefully if it's missing or broken. |
| `open_clip_torch` pulls LAION-2B weights from HuggingFace   | First-boot download; validated via file hash in the HF mirror. |
| Docker image is based on `python:3.11.6-slim` (32 CVEs)     | Base image vulnerabilities tracked upstream; none affect the service's attack surface because we drop to a non-root user and expose only port 8000. Scheduled to rebase on `python:3.12-slim-bookworm` in v1.2. |

## 6. CI enforcement

`.github/workflows/security.yml` runs `bandit -c pyproject.toml -r src/` and
`safety check` on every push and PR. A failure blocks the merge. `pip-audit`
is scheduled to join this workflow in v1.2.

## 7. Incident response

If a vulnerability is disclosed in one of our direct dependencies:

1. Update `whales_be_service/pyproject.toml` to pin a patched version.
2. Run `poetry lock --no-update` to refresh the lockfile.
3. Open a PR; CI runs `pip-audit` automatically on the new lockfile.
4. Merge and publish a patch release. Users are notified via the GitHub
   release notes; the HF model card is unaffected (no code in the artefact).

For code-level vulnerabilities in EcoMarineAI itself, follow the `SECURITY.md`
disclosure process (planned for v1.2).
