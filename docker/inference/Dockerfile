# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock* ./
# Disable native CPU feature detection so llama-cpp-python compiles to a safe ARMv8-A
# baseline. Without this, QEMU exposes a richer feature set than the real Pi 5
# (Cortex-A76, ARMv8.2-A), causing SIGILL (exit 132) when the binary runs on the node.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_SVE=OFF"
# Cache mount: llama-cpp-python and heavy deps downloaded once, reused every build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra inference
FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY docker/inference/server.py ./server.py
COPY src/adapters/inference.py ./adapters/inference.py
COPY src/adapters/prompt.py ./adapters/prompt.py
COPY src/adapters/storage/s3.py ./adapters/storage/s3.py
COPY src/domain/ ./domain/
ENV PATH="/app/.venv/bin:$PATH"
ENV GGUF_PATH=""
ARG IMAGE_SHA=unknown
ENV IMAGE_SHA=${IMAGE_SHA}
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
