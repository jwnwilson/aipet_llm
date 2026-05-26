# syntax=docker/dockerfile:1.7
# Export image — llama.cpp (compiled) + Python ML stack for HF→GGUF conversion.
#
# Why a separate image?
#   The Temporal worker and training image do NOT have llama.cpp installed.
#   GGUF export requires two tools from the llama.cpp repository:
#     1. convert_hf_to_gguf.py  — converts HF checkpoint to FP16 GGUF
#     2. build/bin/llama-quantize — quantises FP16 → Q4_K_M (or other)
#   This image pre-builds both at image-build time so the Job container
#   starts instantly with no build step at runtime.
#
# Environment variables expected at runtime (set by K8sTrainingAdapter):
#   RUN_ID, CHECKPOINT_S3_PREFIX, GGUF_S3_KEY, QUANTIZE,
#   AWS_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

# ── Stage 1: build llama.cpp binaries ──────────────────────────────────────
FROM python:3.12-slim AS llama-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone at a pinned tag for reproducibility; update the tag when bumping.
ARG LLAMA_CPP_TAG=b3447
RUN git clone --depth=1 --branch ${LLAMA_CPP_TAG} \
        https://github.com/ggerganov/llama.cpp.git /llama.cpp \
    || git clone --depth=1 https://github.com/ggerganov/llama.cpp.git /llama.cpp

# Build only the quantize binary to keep image size small.
RUN cmake -S /llama.cpp -B /llama.cpp/build \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /llama.cpp/build --config Release --target llama-quantize -j$(nproc)

# ── Stage 2: build Python virtualenv ───────────────────────────────────────
FROM python:3.12-slim AS py-builder
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock* ./
# Install base deps (includes boto3) + training extra (torch, transformers,
# sentencepiece, safetensors — required by convert_hf_to_gguf.py).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra training

# ── Stage 3: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# llama.cpp: Python scripts (convert_hf_to_gguf.py) + compiled binary
COPY --from=llama-builder /llama.cpp /llama.cpp

# Python virtualenv
COPY --from=py-builder /app/.venv ./.venv

# Application source
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src
# Tells domain.train.export where to find llama.cpp tools at runtime.
ENV LLAMA_CPP_DIR=/llama.cpp

CMD ["python", "-m", "interactors.cli.training.k8s_export"]
