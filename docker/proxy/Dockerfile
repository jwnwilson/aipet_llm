# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock* README* ./
# BuildKit cache mount — pip/uv packages survive across rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
# Pre-build the project wheel so remote adapters (kaggle, runpod, vastai)
# can stage it without needing build tools at runtime.
COPY src/ ./src/
RUN uv build --wheel --out-dir /app/dist/

FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/dist/ ./dist/
COPY src/ ./src/
COPY alembic.ini* ./
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "interactors.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
