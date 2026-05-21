# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock* ./
# Cache mount: llama-cpp-python and heavy deps downloaded once, reused every build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY docker/inference/server.py ./server.py
COPY src/adapters/inference.py ./adapters/inference.py
COPY src/adapters/prompt.py ./adapters/prompt.py
COPY src/domain/ ./domain/
ENV PATH="/app/.venv/bin:$PATH"
ENV GGUF_PATH=""
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
