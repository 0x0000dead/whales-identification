# -------- build stage --------
FROM python:3.11.6-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libssl-dev \
      libffi-dev \
      python3-dev \
      curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv — fast Python package manager (~10x faster than pip/poetry)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# -------- runtime stage --------
FROM python:3.11.6-slim

# Runtime-only system dependencies (OpenCV, libGL for rembg, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
      curl \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src ./src

# Resources directory holds database.csv and species_map.csv; configs holds
# calibrated anti-fraud thresholds. Created empty — docker-entrypoint.sh
# populates them from Hugging Face on first boot.
RUN mkdir -p src/whales_be_service/resources \
 && mkdir -p src/whales_be_service/models \
 && mkdir -p src/whales_be_service/configs \
 && mkdir -p reports

# Entrypoint that fetches model weights from HF on first boot if missing,
# then launches uvicorn. Makes `docker compose up` Just Work without any
# host-side `scripts/download_models.sh` dance.
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create non-root user and grant write access to the dirs the entrypoint
# downloads into.
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
 && chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ALLOWED_ORIGINS="*" \
    HF_REPO="0x0000dead/ecomarineai-cetacean-effb4" \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface

RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache
USER appuser

# Pre-download CLIP weights at build time (file download only — no torch ops,
# safe under QEMU amd64 emulation on ARM).  open_clip.create_model_and_transforms()
# finds the cached file in HF_HOME at container startup and skips the 350 MB download.
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
p = hf_hub_download( \
    'laion/CLIP-ViT-B-32-laion2B-s34B-b79K', \
    'open_clip_pytorch_model.bin' \
); \
print('CLIP pre-cached OK ->', p)"

# Bake ML inference artefacts into the image so docker-entrypoint.sh
# need_download() finds them on first boot and skips the HF download entirely.
# Result: cold-start time drops from ~10 min (network) to ~45 s (disk load).
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
REPO = '0x0000dead/ecomarineai-cetacean-effb4'; \
files = [ \
    ('efficientnet_b4_512_fold0.ckpt', '/app/src/whales_be_service/models'), \
    ('encoder_classes.npy',            '/app/src/whales_be_service/models'), \
    ('species_map.csv',                '/app/src/whales_be_service/resources'), \
    ('anti_fraud_threshold.yaml',      '/app/src/whales_be_service/configs'), \
    ('metrics_baseline.json',          '/app/reports'), \
]; \
[hf_hub_download(REPO, f, local_dir=d, local_dir_use_symlinks=False) \
 or print('  OK', f) for f, d in files]; \
print('ML artefacts baked OK')"

# Pre-download rembg U2-Net model (~176 MB) so the first /predict-single
# with generate_mask=True does not block the worker for minutes.
RUN mkdir -p /app/.u2net && python3 -c "\
import urllib.request, os; \
url = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx'; \
dst = '/app/.u2net/u2net.onnx'; \
urllib.request.urlretrieve(url, dst); \
print('rembg u2net.onnx OK ->', os.path.getsize(dst), 'bytes')"

EXPOSE 8000

# Models are baked into the image — no first-boot download.
# 30 s is enough for disk-based CLIP + EfficientNet load with background warmup.
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
