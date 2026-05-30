# llm-api

LLM training and hosting service. Train lightweight models on a Raspberry Pi 5 k8s cluster or on 3rd party services like kaggle, runpod and vastai, manage them via a React web UI, and expose them for inference via a per-user API key.

The web UI (`ui/`) lets users log in with Auth0, upload training and eval datasets, trigger training runs on remote GPU backends (RunPod, Vast.ai, Kaggle, SSH), monitor pipeline progress, review eval metrics, and manage inference instances. New users require admin approval before accessing the app.

The first supported model type is an AI pet companion — takes a 3D scene + pet stats and returns a valid action + target object.

## Stack

- Python 3.12+, `uv`, FastAPI + uvicorn
- Inference: llama-cpp-python (GGUF, no GPU, ARM64) or OpenRouter
- Training: HuggingFace Transformers + Datasets + PyTorch (dev dep)
- Orchestration: Temporal
- Auth: Auth0 JWT
- Storage: AWS S3
- Compute backends: RunPod, Vast.ai, Kaggle, SSH
- Tests: pytest + httpx + pytest-asyncio

## Quick start

```bash
uv sync
make serve          # uvicorn on :8000 with hot-reload
make request        # POST a test /infer request
```

## Architecture

Three-layer hexagonal design:

| Layer | Path | Role |
|-------|------|------|
| Interactors | `src/interactors/` | Entry points — wire adapters + domain, no business logic |
| Domain | `src/domain/` | Pure business logic, no I/O |
| Adapters | `src/adapters/` | Concrete port implementations (DB, storage, compute, auth) |

### S3 path structure

| Prefix | Contents |
|--------|----------|
| `workflow/{run_id}/` | Per-run artefacts: status, logs, checkpoint, data, GGUF |
| `model/{model_id}.gguf` | Named GGUF exports |
| `dataset/{dataset_id}/` | Shared datasets (`train.jsonl`, `eval.jsonl`) |

`run_id` is always a UUID hex string. Adapters must not prefix it with backend name.

## Training pipeline

Run the full dataset → train → evaluate → export lifecycle as a Temporal workflow.

```bash
# Start infrastructure
docker compose up temporal -d        # Temporal + web UI on :8233
docker compose up temporal-worker    # activity worker

# Trigger a run (via API or CLI)
python -m src.cli.trigger_training --experiment-name run-001 --epochs 10 --patience 3
python -m src.cli.trigger_training --experiment-name sweep-lr --epochs 5 --skip-generate
```

Pipeline stages:

| Stage | Activity | Timeout | Retries |
|-------|----------|---------|---------|
| Generate dataset | `generate_dataset_activity` | 30 min | 3 |
| Fine-tune | `train_activity` | 6 h | 1 |
| Evaluate | `evaluate_activity` | 30 min | 3 |
| Export GGUF | `export_activity` | 1 h | 1 (only if eval passes) |

Workflow UI: http://localhost:8233

## Manual training commands

```bash
make data                      # generate 2000 train + 200 eval examples
make train                     # fine-tune SmolLM-360M
make train DRY_RUN=1           # 1-step smoke test
make evaluate                  # score HF checkpoint (target: ≥ 95% parse rate)
make setup-llama               # clone + build llama.cpp
make export                    # convert checkpoint → models/model.gguf (Q4_K_M)
make evaluate-gguf             # score the GGUF model
```

## CI/CD

Triggers on merge to `main`: builds ARM64 image, pushes to ECR, applies k8s manifests.

### Required secrets

| Secret | Description |
|--------|-------------|
| `AWS_ROLE_ARN` | IAM role for OIDC — `terraform -chdir=infra/terraform output -raw github_actions_role_arn` |
| `AWS_S3_BUCKET` | S3 bucket for training artefacts |
| `LLM_API_AWS_ACCESS_KEY_ID` | AWS access key for the llm-api service account |
| `LLM_API_AWS_SECRET_ACCESS_KEY` | AWS secret key for the llm-api service account |
| `AUTH0_DOMAIN` | Auth0 tenant domain |
| `AUTH0_AUDIENCE` | Auth0 API audience |
| `AUTH0_CLIENT_ID` | Auth0 application client ID |
| `AUTH0_MGMT_CLIENT_ID` | Auth0 M2M app client ID |
| `AUTH0_MGMT_CLIENT_SECRET` | Auth0 M2M app client secret |
| `CORS_ORIGINS` | Comma-separated allowed origins |
| `KUBE_CONFIG` | Base64-encoded kubeconfig — `base64 -i ~/.kube/config.yaml` |

### Seeding secrets for a new repo

```bash
cp .env.example .env            # fill in values
./scripts/set_github_secrets.sh
# AWS_ROLE_ARN and KUBE_CONFIG must be set manually:
gh secret set AWS_ROLE_ARN --body "arn:aws:iam::123456789:role/your-role"
gh secret set KUBE_CONFIG < <(base64 -i ~/.kube/config.yaml)
gh secret list                  # verify
gh workflow run deploy.yml      # trigger deploy
```

## Deployment (Raspberry Pi 5)

### Prerequisites

```bash
docker buildx create --use
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

SSH access to the RPi as `pi` (default: `raspberrypi.local`).

### Deploy

```bash
make docker-deploy RPI_HOST=raspberrypi.local

# Step by step:
make docker-build               # build linux/arm64 image
make docker-export              # save as llm-api.tar.gz
scp llm-api.tar.gz pi@raspberrypi.local:~/

# On the RPi:
docker load -i ~/llm-api.tar.gz
docker compose up -d
```

### Verify

```bash
curl http://raspberrypi.local:8000/health
```

### Hot-swap model without rebuilding

`models/` is a mounted volume:

```bash
scp models/model.gguf pi@raspberrypi.local:~/models/
ssh pi@raspberrypi.local "docker compose restart"
```

## Web UI (`ui/`)

React + TypeScript frontend served alongside the API.

**Stack:** Vite, React 19, React Router, TanStack Query, Auth0, Tailwind CSS, Radix UI, Zod

**Pages:**

| Route | Page |
|-------|------|
| `/models` | List and create models |
| `/models/:id` | Model detail — trigger runs, upload datasets, run inference |
| `/datasets` | Upload and manage training/eval datasets |
| `/runs` | List all runs with status |
| `/runs/:id` | Run detail — pipeline stages, logs, eval metrics |
| `/inferences` | Manage inference instances (start/stop, status) |
| `/admin/users` | User approval (admin only) |

Auth0 login is required. New users land on an "access pending" screen until approved by an admin. In local dev (`APP_ENV=development`) auth is bypassed entirely.

### Run the UI locally

```bash
cd ui
npm install
npm run dev             # Vite dev server on :5173
npm test                # Vitest unit tests
npm run test:coverage   # coverage report
npm run build           # production build → ui/dist/
```

The UI expects the API at `http://localhost:8000` by default. Set `VITE_API_BASE_URL` to override.

### Auth0 environment variables (UI)

| Variable | Description |
|----------|-------------|
| `VITE_AUTH0_DOMAIN` | Auth0 tenant domain |
| `VITE_AUTH0_CLIENT_ID` | Auth0 SPA client ID |
| `VITE_AUTH0_AUDIENCE` | Auth0 API audience |

Copy `ui/.env.example` to `ui/.env.local` and fill in values for local development.

---

## Development

```bash
make help              # list all make targets
uv run pytest          # run tests
uv run pytest tests/unit/
uv run pytest tests/integration/
```

Local dev auth is bypassed when `APP_ENV=development` (uses `FakeAuthAdapter`).
