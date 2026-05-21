# Completed Work

> Archived from plan.md. All items below have corresponding implementation files.

---

## Phase 1: Foundation

### Task 1.1 — Project structure & dependencies
Hexagonal directory layout established; all packages installed via `uv`.
**Outputs:** `pyproject.toml`, `src/` tree with `__init__.py` files.

### Task 1.2 — Pydantic schemas
All input/output data contracts defined.
**Outputs:** `src/domain/models.py`, `src/domain/actions.py`

### Task 1.3 — Domain ports (interfaces)
Abstract `InferencePort` interface defined with unit tests.
**Outputs:** `src/domain/ports.py`, `tests/unit/test_ports.py`

---

## Phase 2: Core Implementation

### Task 2.1 — Inference adapter (llama.cpp)
`LlamaCppInferenceAdapter` implemented; parse failures return `Action.IDLE`.
**Outputs:** `src/infrastructure/inference.py`

### Task 2.2 — Prompt template & response parser
`build_prompt()` and `parse_response()` implemented; prompt stays under 300 tokens.
**Outputs:** `src/infrastructure/prompt.py`

### Task 2.3 — Synthetic training dataset generator
5 000 train / 500 eval examples generated in JSONL format.
**Outputs:** `src/domain/train/dataset.py`, `src/cli/generate_dataset.py`, `data/train.jsonl`, `data/eval.jsonl`

---

## Phase 3: API Layer

### Task 3.1 — FastAPI application
`POST /infer` and `GET /health` endpoints wired with DI for `InferencePort`.
**Outputs:** `src/api/app.py`, `src/api/routes.py`

### Task 3.2 — API integration tests
Full request/response cycle verified with stub adapter.
**Outputs:** `tests/integration/test_api.py`

---

## Phase 4: Training Pipeline

### Task 4.1 — Fine-tuning script
HuggingFace `Trainer` fine-tune on prompt+completion pairs; supports `--dry-run`.
**Outputs:** `src/domain/train/trainer.py`, `src/cli/train.py`

### Task 4.2 — Evaluation & export script
Schema-valid response rate measured; GGUF export via `llama.cpp` converter.
**Outputs:** `src/domain/train/evaluate.py`, `src/cli/evaluate.py`, `src/domain/train/export.py`, `src/cli/export.py`

---

## Phase 5: Deployment

### Task 5.1 — Docker deployment config
Multi-arch ARM64 `Dockerfile` and `docker-compose.yml` for RPi 5.
**Outputs:** `Dockerfile`, `docker-compose.yml`

---

## Phase 6: Model Quality Improvements

> Root-cause fixes for EAT/SLEEP bias and wrong target-object selection.

### Task 6.1 — Statistical quality test suite
Per-stat accuracy report and action-distribution histogram; integration tests gating CI.
**Outputs:** `src/domain/train/quality_report.py`, `tests/integration/test_model_quality.py`

### Task 6.2 — Dataset regeneration
Stratified sampling, tick-parity fix, richer multi-target scenes, 5k/500 dataset.
**Outputs:** Updated `src/domain/train/dataset.py`, regenerated `data/train.jsonl`, `data/eval.jsonl`

### Task 6.3 — Prompt engineering improvements
Stats sorted high→low with `(highest)` label; explicit decision rule; objects sorted by distance.
**Outputs:** Updated `src/infrastructure/prompt.py`, updated `tests/unit/test_prompt.py`

### Task 6.4 — Training improvements
Weighted sampler, cosine LR schedule with warmup, early stopping (`--patience`), per-action eval logging, `--base-model` arg (default SmolLM2-1.7B).
**Outputs:** Updated `src/domain/train/trainer.py`, updated `src/cli/train.py`

---

## Post V1 — Completed

### P.2 — Kubernetes deployment
`Deployment`, `Service`, and `HPA` manifests for multi-node cluster.
**Outputs:** `infra/k8s/deployment.yaml`, `infra/k8s/service.yaml`, `infra/k8s/hpa.yaml`

### P.3 — Temporal training pipeline
Full training lifecycle orchestrated as a Temporal workflow with retry semantics.
**Outputs:** `src/temporal/workflows.py`, `src/temporal/activities.py`, `src/temporal/worker.py`, `src/cli/trigger_training.py`

### P.4 — Remote GPU training (Kaggle / SSH)
`KaggleTrainingAdapter` and `SshTrainingAdapter` implement `RemoteTrainingPort`; `train_activity` routes via `--remote-backend`.
**Outputs:** `src/adapters/kaggle_adapter.py`, `src/adapters/ssh_adapter.py`, `src/adapters/notebook_template.ipynb`

### P.5 — ECR Terraform provisioning
ECR repository, lifecycle policy, GitHub OIDC IAM role, and push policy provisioned via Terraform.
**Outputs:** `infra/terraform/main.tf`, `infra/terraform/github_actions.tf`, `infra/terraform/variables.tf`, `infra/terraform/outputs.tf`, `infra/terraform/versions.tf`

---

## EPIC-1: Kaggle Training Pipeline (Operational)

> End-to-end Kaggle GPU training pipeline is running. Validation of model quality (Feature 1.5) remains pending in plan.md.

### Feature 1.1 — Kaggle credentials
`~/.kaggle/kaggle.json` provisioned; `KAGGLE_USERNAME` and `KAGGLE_KEY` set in shell profile.

### Feature 1.2 — Temporal server (local)
`docker-compose.yml` runs `temporal`, `temporal-db`, and `temporal-ui`; Temporal UI at http://localhost:8233.

### Feature 1.3 — Temporal worker (local)
Worker runs outside Docker via `uv run python -m src.temporal.worker`; handles task queue `llm-api-training`.

### Feature 1.4 — Dataset generation and training trigger
- `src/cli/generate_dataset.py` produces 5 000 train / 500 eval examples.
- `src/cli/trigger_training.py` submits the Temporal workflow with `--remote-backend kaggle`.
- `evaluate_activity` and `export_activity` wired end-to-end; GGUF written to `models/model.gguf` only when eval ≥ 95%.
- Async API endpoints added for workflow triggering: `POST /workflows/training`, `POST /workflows/evaluate`, `POST /workflows/export`.
- Alembic migrations in place for workflow run tracking.

---

## EPIC-3: CI/CD Automation

### Feature 3.1 — GitHub Actions deploy pipeline

#### TASK-3.1.1 — `.github/workflows/deploy.yml`
Triggers on successful `Test` workflow run against `main`; OIDC via `secrets.AWS_ROLE_ARN`; builds linux/arm64 image with GHA layer cache; tags `:<sha>` and `:latest`; applies k8s manifests and waits for rollout with `--timeout=600s`.
**Outputs:** `.github/workflows/deploy.yml`

#### TASK-3.1.2 — GitHub secrets seeded
`AWS_ROLE_ARN` and `KUBECONFIG` (and additional secrets for DB, ECR, Auth0, Kaggle, RunPod, Vast) set via `gh secret set` after `terraform apply`.

#### TASK-3.1.3 — k8s deployment uses ECR URL
Deploy pipeline does `sed -i "s|<ECR_REPOSITORY_URL>:latest|$IMAGE|g"` at deploy time; static manifests keep the placeholder intentionally.
**Outputs:** `infra/k8s/llm-api/deployment.yaml`, `infra/k8s/temporal/worker.yaml`

#### TASK-3.1.4 — Terraform state files in `.gitignore`
`.gitignore` entries: `infra/terraform/**/.terraform/`, `infra/terraform/*.tfstate*`, `infra/terraform/**/.terraform.lock.hcl`.

---

## EPIC-4: Production Hardening

### Feature 4.1 — Early stopping verification

#### TASK-4.1.1 — `--patience` smoke-test
`--patience` flag is implemented in `src/interactors/cli/training/train.py`; `EarlyStoppingCallback` wired in trainer. Verified via `uv run python -m src.cli.train --dry-run --patience 1`.

#### TASK-4.1.2 — Training flags documented in `README.md`
`--patience`, `--warmup-ratio`, `--base-model`, and `--remote-backend` documented with example invocations. Auth0 and CORS env vars also documented.
**Outputs:** Updated `README.md`

---

## EPIC-5: Auto Deployment & Model Availability

> Implemented with **AWS S3** instead of GCP GCS. All functionality is equivalent.

### Feature 5.1 — Cloud Storage Adapter (AWS S3)
`S3StorageAdapter` in `src/adapters/storage/s3.py` implements `StoragePort`; uploads/downloads GGUF artifacts keyed by run ID. Config via `AWS_S3_BUCKET` and standard boto3 credential chain.
**Outputs:** `src/adapters/storage/s3.py`, `tests/unit/test_s3_storage.py`

### Feature 5.2 — Upload wired into `export_activity`
After GGUF is written, `export_activity` calls `upload_model()` when `AWS_S3_BUCKET` is set; S3 key logged so the Temporal UI shows the artifact location.
**Outputs:** Updated `src/interactors/temporal/activities.py`

### Feature 5.3 — Model management API endpoints
- `GET /api/models` — list all registered models
- `GET /api/models/{model_id}` — get model by ID
- `POST /api/models/{model_id}/activate` — download GGUF from S3, hot-swap the inference adapter, mark as active
- `POST /api/models` — register a new model record
**Outputs:** `src/interactors/api/routes/models.py`, `tests/integration/test_model_workflow_integration.py`

### Feature 5.4 — Hot-swap support in `LlamaCppInferenceAdapter`
`release()` method unloads the current model from RAM; `activate_model` route acquires new GGUF from S3, calls `release()` on the old adapter, and loads the new one. No explicit lock needed — FastAPI handles request concurrency.
**Outputs:** Updated `src/adapters/inference.py`

---

## EPIC-6: Authentication for Public Access

> Implemented with **Auth0 JWT authentication** instead of static API keys. Provides stronger security and user identity without managing key distribution.

### Feature 6.1 — Auth0 JWT middleware
`Auth0Adapter` in `src/adapters/auth/auth0.py` validates JWTs against the Auth0 JWKS endpoint. `FakeAuthAdapter` in `src/adapters/auth/fake.py` used in local dev (`APP_ENV=development`). `require_auth`, `get_current_user`, `require_approved`, `require_admin` dependencies in `src/interactors/api/auth.py`.
**Outputs:** `src/adapters/auth/auth0.py`, `src/adapters/auth/fake.py`, `src/interactors/api/auth.py`

### Feature 6.2 — Auth applied to all routers
All routers use `require_approved` or `require_admin` as router-level dependency; `GET /health` remains unauthenticated.
**Outputs:** Updated `src/interactors/api/app.py` and all route files

### Feature 6.3 — CORS configured
`CORSMiddleware` reads `CORS_ORIGINS` env var; defaults to `[]` in production, `[localhost:*]` when `APP_ENV=development`.
**Outputs:** Updated `src/interactors/api/app.py`

### Feature 6.4 — Auth integration tests
Full request cycle tested: unauthenticated → 401, invalid token → 401, valid token → 200, `GET /health` → 200 without token.
**Outputs:** `tests/integration/test_auth.py`

---

## EPIC-7: Project Consolidation

> Rename the project to "llm-api" and make it a generic training platform.

### Feature 7.1 — Rename project to llm-api

#### TASK-7.1.1 — Remove legacy branding references
Full aipet → llm-api sweep across `pyproject.toml`, `docker-compose.yml`, k8s manifests, Terraform, source files, and all string literals. Two-round sed sweep + targeted fixups; Terraform state buckets and S3 bucket kept at original names (cannot rename existing resources).
**Outputs:** Updated `pyproject.toml`, `docker-compose.yml`, `infra/k8s/llm-api/`, `infra/terraform/`, source files

#### TASK-7.1.2 — Integrate llm-ui into this repo
React/TypeScript frontend added as `ui/` sub-project (Vite + React + Tailwind). Docker Compose updated to serve the UI; GitHub Actions workflow added for UI deploy.
**Outputs:** `ui/` directory, updated `docker-compose.yml`, `.github/workflows/deploy-ui.yml`

---

## EPIC-8: LLM Training Pipeline

> Improve reliability, observability, and user control over the training pipeline.

### Feature 8.1 — Error handling in workflows

#### TASK-8.1.1 — Update runs to failed/cancelled status with error message
`fail_run` method added to `RunStorePort` and `DatabaseRunStore`; `fail_run_activity` Temporal activity calls it. `RunStatus.CANCELLED` added (distinct from `FAILED`). All three workflows (`TrainingPipelineWorkflow`, `EvaluateWorkflow`, `ExportWorkflow`) wrap their body in `try/except` using `is_cancelled_exception()` to distinguish user cancellations from hard failures; `fail_run_activity` is called in all error paths before re-raising. Unit tests cover failure, cancellation, and default-status paths.
**Outputs:** Updated `src/domain/models.py`, `src/domain/ports.py`, `src/adapters/database/run_store.py`, `src/interactors/temporal/activities.py`, `src/interactors/temporal/workflows.py`, `src/interactors/temporal/worker.py`, `tests/unit/test_temporal_activities.py`, `tests/unit/test_temporal_workflow.py`

### Feature 8.2 — Run overrides flowing to the pipeline

#### TASK-8.2.1 — Run overrides reach the pipeline
`epochs`, `patience`, `warmup_ratio`, `remote_backend`, `base_model`, `num_train_samples`, and `num_eval_samples` all flow from API trigger → `RunConfig.training_config` (persisted as JSON blob) → `ExperimentConfig` → Temporal workflow activities. Override tests present in `test_remote_adapters.py` (epochs, remote_backend). `training_config` column added via migration `0006`.
**Outputs:** Updated `src/interactors/api/routes/runs.py`, `src/domain/models.py`, `src/adapters/database/run_store.py`, `src/adapters/database/alembic/versions/0006_add_model_config_to_runs.py`, updated `ui/src/types/index.ts`, `ui/src/pages/RunDetailPage.tsx`

### Feature 8.3 — User-controlled training via UI

#### TASK-8.3.1 — Upload training and eval datasets via UI
`POST /api/datasets/train` and `POST /api/datasets/eval` endpoints accept multipart JSONL file uploads and store them via `StoragePort`. `write_stream()` added to `StoragePort`. `DatasetUpload` React component (file pickers + upload button + per-upload error display) placed on the Model detail page.
**Outputs:** `src/interactors/api/routes/datasets.py`, updated `src/domain/ports.py`, `ui/src/components/DatasetUpload.tsx`, updated `ui/src/pages/ModelDetailPage.tsx`

#### TASK-8.3.2 — Select base model via UI
"Base model" free-text input added to the RunModal trigger form; value passed as `base_model` in the workflow trigger payload.
**Outputs:** Updated `ui/src/components/RunModal.tsx`

#### TASK-8.3.3 — Select training platform via UI
"Remote backend" dropdown (local / kaggle / ssh / vastai / colab) added to RunModal; value passed as `remote_backend` in the trigger payload.
**Outputs:** Updated `ui/src/components/RunModal.tsx`

### Feature 8.4 — Eval improvements

#### TASK-8.4.1 — Improve eval metrics and expose via API
`QualityReport` Pydantic model captures per-stat accuracy, target-object accuracy, priority-conflict accuracy, fallback accuracy, and action distribution. Written to `data/workflow/{run_id}/quality_report.json` during `evaluate_activity`. `GET /api/runs/{run_id}/evaluation` returns `EvaluationData` (run status + eval score + quality report).
**Outputs:** `src/domain/models.py` (`QualityReport`, `EvaluationData`), updated `src/interactors/temporal/activities.py`, `src/interactors/api/routes/runs.py`

#### TASK-8.4.2 — Display eval results in UI
Eval panel added to Run detail page: fetches `GET /api/runs/{run_id}/evaluation` (enabled only for terminal runs with `eval_valid_pct`), shows pass/fail badge, overall score, per-stat accuracy table, and action distribution. `EvalMetrics` component extended with `qualityReport` prop. Cancel-run button added (calls `POST /api/runs/{run_id}/cancel`); row-click navigation on Model list page.
**Outputs:** Updated `ui/src/pages/RunDetailPage.tsx`, updated `ui/src/components/EvalMetrics.tsx`, `ui/src/api/runs.ts` (`cancelRun`, `isRunCancellable`), updated `src/interactors/api/routes/runs.py`

---

## EPIC-9: Inference Proxy

> llm-api acts as a unified inference proxy, routing requests to either OpenRouter (cloud LLMs) or a locally-hosted GGUF model, selected by model ID at request time.

### TASK-9.1 — OpenRouter inference adapter
`OpenRouterInferenceAdapter` implemented in `src/adapters/inference_openrouter.py`. Converts `InferenceRequest` to OpenRouter chat completion format; parses JSON response back to `InferenceResponse`. Falls back to `Action.IDLE` on parse failure.
**Outputs:** `src/adapters/inference_openrouter.py`, `tests/unit/test_inference_openrouter.py`

### TASK-9.2 — Backend field on model records
`backend: Literal["local", "openrouter"]` and `backend_model_id: str` added to `TrainingModel`. `POST /api/models` accepts and persists these fields; DB migration added.
**Outputs:** Updated `src/domain/models.py`, DB migration, updated `src/interactors/api/routes/models.py`

### TASK-9.3 — Per-model inference endpoint with backend routing
`POST /api/models/{model_id}/infer` dispatches to `OpenRouterInferenceAdapter` for `openrouter` models or `LlamaCppInferenceAdapter` for `local` models. Returns unified `InferenceResponse` regardless of backend.
**Outputs:** New route in `src/interactors/api/routes/models.py`, integration tests

### TASK-9.4 — Lazy load for local GGUF models
Local models loaded on first inference request rather than at startup. Only one GGUF held in memory at a time (RPi 8 GB constraint) — activating a second local model unloads the first. `GET /api/models/{model_id}` exposes `inference_status: unloaded | loading | ready`. `/health` returns 200 immediately regardless of model state.
**Outputs:** Updated `src/adapters/inference.py`, updated `src/interactors/api/app.py`, updated model status in routes

### TASK-9.5 — Inference UI
Inference panel added to the model detail page: structured form / raw JSON input for `InferenceRequest`, "Run inference" button, and response display. Calls `POST /api/models/{model_id}/infer`. Shows which backend (OpenRouter / local) the model uses.
**Outputs:** Updated `ui/src/pages/ModelDetailPage.tsx`, UI inference component
