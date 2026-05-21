# Epic-13: Inference Management — Implementation Plan

## Overview

Epic-13 introduces a first-class **Inference Management** layer to the platform. Users gain a dedicated UI tab listing every inference instance derived from a trained model. Instances can be started (spawning a K8s pod for local models) or stopped, and idle instances auto-terminate after a configurable timeout.

The platform is split into **two Docker containers**: a lightweight proxy API that manages state and routes requests, and a heavy inference worker that loads GGUF models and serves them. Heavy Python packages (torch, llama-cpp-python) are cached at the Docker layer level to avoid re-downloading on every rebuild.

---

## Architecture

```
UI (InferencePage)
    │  GET/POST /api/inferences/*
    ▼
┌─────────────────────────────────────────┐
│  PROXY API CONTAINER  (port 8000)       │
│  FastAPI + SQLAlchemy + K8sPodAdapter   │
│  inference state management             │
│  idle shutdown background task          │
└────────────────┬────────────────────────┘
                 │ forwards /infer to pod URL
                 │ manages K8s pods via K8s API
                 ▼
┌─────────────────────────────────────────┐
│  INFERENCE WORKER CONTAINER  (port 8080)│
│  llama-cpp-python HTTP server           │
│  POST /infer  →  InferenceResponse      │
│  (one pod per InferenceInstance)        │
└─────────────────────────────────────────┘
         ▲  (running as K8s pod)
         │
Kubernetes API

Temporal workflow
    └─ create_inference_activity  (auto-creates record on run completion)

asyncio background task (in proxy)
    └─ idle shutdown loop  (stops instances unused > INFERENCE_IDLE_TIMEOUT_HOURS)
```

---

## Docker Container Design (TASK-13.7)

### Container split rationale

| Concern | Proxy API | Inference Worker |
|---------|-----------|-----------------|
| FastAPI routes + auth | ✅ | ❌ |
| SQLAlchemy / DB access | ✅ | ❌ |
| K8s pod orchestration | ✅ | ❌ |
| llama-cpp-python | ❌ | ✅ |
| torch (future) | ❌ | ✅ |
| Typical image size | ~300 MB | ~3–5 GB |
| Restarts on code change | Fast | Fast (deps cached) |

---

### Proxy API — `docker/proxy/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Cache uv package downloads — persists across rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group proxy
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY src/ ./src/
COPY alembic.ini ./
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "interactors.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

**Key points:**
- Only installs the `proxy` dependency group — no torch, no llama-cpp-python
- BuildKit `--mount=type=cache` on `/root/.cache/uv` persists the package cache across every rebuild
- `--frozen` ensures reproducible installs from `uv.lock`

---

### Inference Worker — `docker/inference/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Heavy packages (torch, llama-cpp-python) downloaded once, reused on
# every subsequent build as long as resolved versions do not change.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group inference
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
```

**Key points:**
- Only installs the `inference` dependency group
- `GGUF_PATH` env var injected by K8s at pod start; container loads that specific model
- `server.py` is a thin FastAPI wrapper around `LlamaCppInferenceAdapter`

---

### Inference Worker HTTP server — `docker/inference/server.py`

```python
import os
from fastapi import FastAPI
from domain.models import InferenceRequest, InferenceResponse
from adapters.inference import LlamaCppInferenceAdapter

app = FastAPI()
_adapter: LlamaCppInferenceAdapter | None = None

@app.on_event("startup")
async def startup() -> None:
    global _adapter
    _adapter = LlamaCppInferenceAdapter(model_path=os.environ["GGUF_PATH"])

@app.get("/health")
def health() -> dict:
    return {"status": "ready", "model": os.environ.get("GGUF_PATH", "")}

@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    assert _adapter is not None
    return _adapter.infer(request)
```

---

### Dependency groups — `pyproject.toml`

```toml
[tool.uv.groups]
proxy = [
  "fastapi", "uvicorn[standard]", "sqlalchemy", "alembic",
  "httpx", "pydantic", "kubernetes", "slowapi", "auth0-python",
]
inference = [
  "llama-cpp-python", "fastapi", "uvicorn[standard]", "pydantic",
  # torch added here when needed for quantisation/eval inside container
]
```

---

### Local development — `docker-compose.yml`

```yaml
version: "3.9"
services:
  proxy:
    build: { context: ., dockerfile: docker/proxy/Dockerfile }
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql://llm:llm@db:5432/llm
      K8S_MOCK: "true"
      INFERENCE_WORKER_URL: http://inference:8080
    depends_on: [db]
  inference:
    build: { context: ., dockerfile: docker/inference/Dockerfile }
    ports: ["8080:8080"]
    volumes: ["./models:/models:ro"]
    environment:
      GGUF_PATH: /models/model.gguf
  db:
    image: postgres:16-alpine
    environment: { POSTGRES_USER: llm, POSTGRES_PASSWORD: llm, POSTGRES_DB: llm }
    volumes: [pgdata:/var/lib/postgresql/data]
volumes:
  pgdata:
```

---

### How the proxy forwards inference to the worker

1. Receives `POST /api/inferences/{id}/infer`
2. Looks up `InferenceInstance` — asserts `status=ready`
3. Stamps `last_used_at = now()`
4. Resolves worker URL: `http://{pod_name}.{K8S_NAMESPACE}.svc.cluster.local:8080/infer`
   (or `INFERENCE_WORKER_URL` env var in local dev)
5. Forwards request body via `httpx`; returns response

The proxy is stateless with respect to model weights — all model loading lives in the worker.

---

