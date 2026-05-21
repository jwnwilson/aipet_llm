# Plan

> Completed work → [complete.md](complete.md)

---

## EPIC-9: Inference Proxy

> llm-api acts as a unified inference proxy, routing requests to either OpenRouter (cloud LLMs) or a locally-hosted GGUF model, selected by model ID at request time.

### TASK-9.1 — OpenRouter inference adapter
Implement `OpenRouterInferenceAdapter` in `src/adapters/inference_openrouter.py` implementing `InferencePort`. Configured via `OPENROUTER_API_KEY` env var. Converts `InferenceRequest` to the OpenRouter chat completion format and parses the JSON response back to `InferenceResponse`. Falls back to `Action.IDLE` on parse failure (consistent with existing adapter contract).

**Outputs:** `src/adapters/inference_openrouter.py`, `tests/unit/test_inference_openrouter.py`

### TASK-9.2 — Backend field on model records
Add `backend: Literal["local", "openrouter"]` and `backend_model_id: str` to the `Model` domain model. `backend_model_id` holds the OpenRouter model string (e.g. `"anthropic/claude-3-haiku"`) for cloud models, or the GGUF path for local models. Update `POST /api/models` to accept and persist these fields; add a DB migration.

**Outputs:** Updated `src/domain/models.py`, DB migration, updated `src/interactors/api/routes/models.py`

### TASK-9.3 — Per-model inference endpoint with backend routing
Add `POST /api/models/{model_id}/infer` that dispatches to the model's configured backend — `OpenRouterInferenceAdapter` for `openrouter` models, `LlamaCppInferenceAdapter` for `local` models. Returns a unified `InferenceResponse` regardless of backend.

**Outputs:** New route in `src/interactors/api/routes/models.py`, integration tests

### TASK-9.4 — Lazy load for local GGUF models
Local models are loaded on first inference request rather than at startup. Only one GGUF is held in memory at a time (RPi 8 GB constraint) — activating a second local model unloads the first. `GET /api/models/{model_id}` exposes `status: unloaded | loading | ready`. `/health` returns 200 immediately regardless of model state.

**Outputs:** Updated `src/adapters/inference.py`, updated `src/interactors/api/app.py`, updated model status in routes

### TASK-9.5 — Inference UI
Add an inference panel to the model detail page: a structured form or raw JSON input for `InferenceRequest`, a "Run inference" button, and a response display. Calls `POST /api/models/{model_id}/infer`. Show which backend (OpenRouter / local) the model uses.

**Outputs:** Updated `ui/src/pages/ModelDetailPage.tsx`, UI inference component

---

## EPIC-10: LLM API — API Keys & Rate Limiting

> Per-user API keys and rate limiting for the inference endpoints.

### TASK-10.1 — Issue API keys per Auth0 user
Add `POST /api/keys` (create key) and `GET /api/keys` (list user's keys) endpoints. Store hashed keys in the DB linked to the Auth0 user ID. Keys are presented once on creation.

**Outputs:** `src/interactors/api/routes/keys.py`, `src/adapters/database/key_store.py`, DB migration

### TASK-10.2 — Accept API key as `Authorization: Bearer` on inference endpoints
Allow inference endpoints to authenticate via either a JWT (Auth0) or a raw API key. Add key lookup to the `require_auth` dependency path.

**Outputs:** Updated `src/interactors/api/auth.py`

### TASK-10.3 — Rate limit inference requests per user
Add rate limiting middleware (e.g. `slowapi`) to cap inference requests per user per minute. Return HTTP 429 with `Retry-After` when the limit is exceeded.

**Outputs:** Updated `src/interactors/api/app.py`, new rate limit config

---

## EPIC-11: Fast E2E Tests

> Re-enable the E2E test suite on CI/CD without slowing down every PR.

### TASK-11.1 — Add scheduled E2E workflow
Add `.github/workflows/e2e.yml` triggered on `schedule: cron` (once daily at 02:00 UTC) and on `workflow_dispatch`. Run `pytest tests/e2e/` against the deployed environment with appropriate secrets.

**Outputs:** `.github/workflows/e2e.yml`

### TASK-11.2 — Fix or skip currently broken E2E tests
Audit `tests/e2e/` and either fix broken tests or mark them `@pytest.mark.skip(reason="...")` with a tracking note. Ensure the suite passes cleanly in the scheduled run.

**Outputs:** Updated `tests/e2e/` files

---

## Epic-12 - Improve dataset management
Datasets have a 1-many relationship with models, ensure that we can support that in the API. Add a new tab in the UI for datasets.

We want to ability to:
- Upload a new training and eval dataset in the dataset tab
- Select a dataset when starting a run with a model
- Track which dataset has been used for a run and show that for the run on the ui

---

## Epic-13 - Add inference management to ui

> Users can browse trained models in a dedicated Inference tab, start/stop inference instances, and track status. Local models run as K8s pods; OpenRouter models go live immediately. Idle instances auto-terminate after a configurable timeout.

### TASK-13.1 — Inference instance domain model & database store

Add `InferenceStatus` enum (`unloaded | initialising | ready | error | terminated`) and `InferenceConfig` / `InferenceInstance` Pydantic models to `src/domain/models.py`. Add `InferenceStorePort` abstract CRUD port to `src/domain/ports.py`. Implement `SQLAlchemyInferenceStore` adapter and an Alembic migration that creates the `inference_instances` table (columns: `id`, `model_id`, `run_id`, `backend`, `backend_model_id`, `gguf_path`, `status`, `pod_name`, `last_used_at`, `idle_timeout_hours`, `created_at`, `updated_at`).

**Outputs:** `src/domain/models.py`, `src/domain/ports.py`, `src/adapters/database/inference_store.py`, new Alembic migration, `tests/unit/test_inference_store.py`

### TASK-13.2 — Inference management REST API

Add `src/interactors/api/routes/inferences.py` with the following endpoints (all require `require_approved`):
- `GET /api/inferences` — list all instances
- `POST /api/inferences` — create instance from `{model_id, run_id?}`
- `GET /api/inferences/{id}` — get details + current status
- `POST /api/inferences/{id}/start` — provision & start (triggers K8s pod for `local`; sets `ready` for `openrouter`)
- `POST /api/inferences/{id}/stop` — teardown (terminates pod for `local`; sets `terminated` for `openrouter`)
- `DELETE /api/inferences/{id}` — delete record (only if `unloaded` or `terminated`)

Register the router in `src/interactors/api/app.py`. Add integration tests.

**Outputs:** `src/interactors/api/routes/inferences.py`, updated `src/interactors/api/app.py`, `tests/integration/test_inferences_api.py`

### TASK-13.3 — Auto-create inference instance on run completion

After the export activity succeeds in the Temporal workflow, add a new activity `create_inference_activity` that auto-creates an `InferenceInstance` record (`status=unloaded`) linked to the completed run's model and GGUF path. Uses `InferenceStorePort` injected via `deps.py`.

**Outputs:** Updated `src/interactors/temporal/activities.py`, updated `src/interactors/temporal/workflows.py`, `tests/unit/test_create_inference_activity.py`

### TASK-13.4 — Kubernetes pod adapter for local model serving

Implement `K8sPodAdapter` in `src/adapters/compute/k8s.py` with `start(instance: InferenceInstance) -> str` (returns pod name) and `stop(pod_name: str) -> None`. The pod spec mounts the GGUF from storage and serves it via `llama-cpp-python` HTTP server. Poll K8s API for pod phase (`Pending → Running → Succeeded/Failed`) and update `InferenceInstance.status` accordingly. The `start` endpoint for `backend=openrouter` skips K8s and immediately sets `status=ready`. Configured via `KUBECONFIG` / `K8S_NAMESPACE` env vars.

**Outputs:** `src/adapters/compute/k8s.py`, updated `src/interactors/api/routes/inferences.py`, `tests/unit/test_k8s_adapter.py`

### TASK-13.5 — Idle inference shutdown background task

Add an `asyncio` background task (started in `app.lifespan`) that polls all `ready` `InferenceInstance` records every 5 minutes. Any instance whose `last_used_at` is older than `INFERENCE_IDLE_TIMEOUT_HOURS` (env var, default `2`) is stopped via the same logic as `POST /api/inferences/{id}/stop`. Update `last_used_at` on every successful `/infer` call. Expose current timeout value via `GET /health`.

**Outputs:** Updated `src/interactors/api/app.py`, updated inference routes (stamp `last_used_at`), `tests/unit/test_idle_shutdown.py`

### TASK-13.6 — Inference management UI tab

Add a new **Inference** tab to the main navigation.

- `ui/src/api/inferences.ts` — typed API client (`listInferences`, `createInference`, `startInference`, `stopInference`, `deleteInference`)
- `ui/src/types/index.ts` — add `InferenceStatus`, `InferenceInstance` TypeScript types
- `ui/src/pages/InferencePage.tsx` — table of all instances with columns: Model name, Backend, Status badge, Last used, Actions (Start / Stop / Delete). Empty state when no instances exist.
- `ui/src/components/InferenceStatusBadge.tsx` — coloured badge for each `InferenceStatus`

Wire the route in `App.tsx` / router config.

**Outputs:** `ui/src/api/inferences.ts`, `ui/src/pages/InferencePage.tsx`, `ui/src/components/InferenceStatusBadge.tsx`, updated `ui/src/types/index.ts`, updated router/nav

---

## Epic-14 - Per User data
I want to filter models, datasets and runs by user so that users can have their own private models. Update the database so that all our data can have an owner, then add filters to the API to filter responses by the user. Ensure this is done from the auth data / jwt signature and automatically applied to avoid users from seeing other users data. 