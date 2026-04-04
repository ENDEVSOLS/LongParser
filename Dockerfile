# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS runtime

# Get uv binaries (fast dependency sync)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# System deps (OCR + MIME sniffing + curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libmagic1 \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install deps with good caching:
# 1) copy lockfiles + local deps first (layer cached unless deps change)
COPY pyproject.toml uv.lock ./


# 2) install only dependencies (not project) — cache-friendly
# Use --frozen to respect lockfile, skip CUDA/NVIDIA packages (installed as CPU-only later)
ENV UV_HTTP_TIMEOUT=300
RUN uv sync --no-cache --frozen --no-install-project --extra api --extra embeddings --extra chroma --extra latex-ocr \
    --no-install-package torch \
    --no-install-package torchvision \
    --no-install-package nvidia-cublas-cu12 \
    --no-install-package nvidia-cuda-cupti-cu12 \
    --no-install-package nvidia-cuda-nvrtc-cu12 \
    --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cudnn-cu12 \
    --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-cufile-cu12 \
    --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-cusolver-cu12 \
    --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-cusparselt-cu12 \
    --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-nvjitlink-cu12 \
    --no-install-package nvidia-nvshmem-cu12 \
    --no-install-package nvidia-nvtx-cu12 \
    --no-install-package triton \
    --no-install-package cuda-bindings \
    --no-install-package cuda-core

# 3) copy source
COPY . .

# 4) install the project itself (skip torch/CUDA, installed as CPU-only next)
RUN uv sync --no-cache --frozen --extra api --extra embeddings --extra chroma --extra latex-ocr \
    --no-install-package torch \
    --no-install-package torchvision \
    --no-install-package nvidia-cublas-cu12 \
    --no-install-package nvidia-cuda-cupti-cu12 \
    --no-install-package nvidia-cuda-nvrtc-cu12 \
    --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cudnn-cu12 \
    --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-cufile-cu12 \
    --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-cusolver-cu12 \
    --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-cusparselt-cu12 \
    --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-nvjitlink-cu12 \
    --no-install-package nvidia-nvshmem-cu12 \
    --no-install-package nvidia-nvtx-cu12 \
    --no-install-package triton \
    --no-install-package cuda-bindings \
    --no-install-package cuda-core

# 5) Install CPU-only PyTorch (MUST be after uv sync to avoid being wiped)
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6) Pre-download pix2tex model weights so they're baked into the image
RUN .venv/bin/python -c "from pix2tex.cli import LatexOCR; LatexOCR()"

# Non-root user (security)
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD [".venv/bin/uvicorn", "clean_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
