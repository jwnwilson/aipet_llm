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
We want users to be able to test new models that they train on the platform. Add an inference tab that will show a list of trained models that are available. Create inference table and model in the API to store models + datasets + config to load for the user. A new record will be created for each successfully trained model and the model will track if the inference is available and laoded or needs to be initialised.

For local models that are not using openrouter should be able to be started on our k8 cluster. Setup logic so that the backend can trigger a new pod for each desired inference to be avilable. We will also need to track the state of the pod so the user knows when it's available. We will need a cron job to shut down inference that have not been used for 2 hours. This timelimit should be configurable.

---

## Epic-14 - Per User data
I want to filter models, datasets and runs by user so that users can have their own private models. Update the database so that all our data can have an owner, then add filters to the API to filter responses by the user. Ensure this is done from the auth data / jwt signature and automatically applied to avoid users from seeing other users data. 