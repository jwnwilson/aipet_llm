# Docker Infrastructure

This project ships two Docker images and a Postgres service orchestrated by `docker-compose.yml`.

## Containers

### `proxy` (port 8000)
Lightweight FastAPI service — handles HTTP routing, authentication, dataset management, and model store
operations. Does **not** include torch or llama-cpp-python, so the image stays small (~300 MB).

Built from `docker/proxy/Dockerfile`.

### `inference` (port 8080)
Heavy worker that loads a GGUF model via `llama-cpp-python` and serves a single `/infer` endpoint.
Kept separate so the proxy can be deployed independently (or on a resource-constrained node) while
inference workers scale out on dedicated hardware.

Built from `docker/inference/Dockerfile`.

### `db`
Postgres 16 (Alpine). Used by the proxy for model/run metadata. Data is persisted in the named
volume `pgdata`.

## Building with BuildKit

Enable BuildKit for layer caching (uv/pip downloads are cached across rebuilds):

```bash
DOCKER_BUILDKIT=1 docker compose build
```

Or set it permanently in your environment:

```bash
export DOCKER_BUILDKIT=1
```

## Running Locally

```bash
# Start all services
docker compose up

# Start only the proxy + db (no inference worker)
docker compose up proxy db

# Rebuild a single service after code changes
docker compose build proxy && docker compose up proxy
```

## Environment Variables

### proxy

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `K8S_MOCK` | `"true"` | Mock Kubernetes backend when `true` |
| `INFERENCE_WORKER_URL` | `http://inference:8080` | Base URL of the inference worker |

### inference

| Variable | Default | Description |
|---|---|---|
| `GGUF_PATH` | `""` | Path to the `.gguf` model file inside the container |

Mount your GGUF model into the container using the `./models` volume:

```bash
cp models/model.gguf ./models/model.gguf
docker compose up inference
```

## Health Checks

- Proxy: `GET http://localhost:8000/health`
- Inference worker: `GET http://localhost:8080/health`
