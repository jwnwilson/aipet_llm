# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra training
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY src/ ./src/
COPY alembic.ini* ./
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "interactors.temporal.worker"]
