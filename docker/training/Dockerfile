# syntax=docker/dockerfile:1.7
# Training image — torch/transformers for fine-tuning on k8s batch Jobs.
# Does NOT include llama-cpp-python (that lives in the inference image).
# Built for linux/arm64 to match the Raspberry Pi 5 cluster.
#
# Environment variables expected at runtime (set by K8sTrainingAdapter):
#   JOB_TYPE=train, RUN_ID, TRAIN_DATA_KEY, EVAL_DATA_KEY, MODEL,
#   EPOCHS (optional), PATIENCE (optional), WARMUP_RATIO (optional),
#   AWS_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
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
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "interactors.cli.training.remote_worker"]
