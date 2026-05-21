# Epic-13: Inference Management — Implementation Plan

## Overview

Epic-13 introduces a first-class **Inference Management** layer to the platform. Users gain a dedicated UI tab listing every inference instance derived from a trained model. Instances can be started (spawning a K8s pod for local models) or stopped, and idle instances auto-terminate after a configurable timeout.

---

## Architecture

```
UI (InferencePage)
    │  GET/POST /api/inferences/*
    ▼
FastAPI routes (inferences.py)
    │  InferenceStorePort
    ▼
SQLAlchemyInferenceStore ──► inference_instances table
    │  K8sPodAdapter (local models)
    ▼
Kubernetes API ──► llama-cpp-python HTTP pod

Temporal workflow
    └─ create_inference_activity  (auto-creates record on run completion)

asyncio background task
    └─ idle shutdown loop  (stops instances unused > INFERENCE_IDLE_TIMEOUT_HOURS)
```

---

## Data Model

### `InferenceStatus` (enum)

| Value          | Meaning                                      |
|----------------|----------------------------------------------|
| `unloaded`     | Record exists; no pod/process running        |
| `initialising` | Pod is starting / GGUF loading               |
| `ready`        | Serving inference requests                   |
| `error`        | Start failed; pod in error state             |
| `terminated`   | Explicitly stopped or timed out              |

### `InferenceConfig` (create/update payload)

```python
class InferenceConfig(BaseModel):
    model_id: str
    run_id: str | None = None          # links to the originating run
    backend: Literal["local", "openrouter"]
    backend_model_id: str = ""         # OpenRouter model string, empty for local
    gguf_path: str = ""                # storage key for the GGUF file
    idle_timeout_hours: float = 2.0    # env: INFERENCE_IDLE_TIMEOUT_HOURS
```

### `InferenceInstance` (full record)

```python
class InferenceInstance(InferenceConfig):
    id: str
    status: InferenceStatus = InferenceStatus.UNLOADED
    pod_name: str | None = None        # K8s pod name (local only)
    last_used_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
```

### DB migration — `inference_instances` table

| Column               | Type      | Notes                                  |
|----------------------|-----------|----------------------------------------|
| `id`                 | UUID PK   |                                        |
| `model_id`           | VARCHAR   | FK → training_models.id (soft ref)     |
| `run_id`             | VARCHAR   | nullable; FK → training_runs.id        |
| `backend`            | VARCHAR   | `local` or `openrouter`                |
| `backend_model_id`   | VARCHAR   | OpenRouter model string                |
| `gguf_path`          | VARCHAR   | storage key for GGUF                   |
| `status`             | VARCHAR   | InferenceStatus value                  |
| `pod_name`           | VARCHAR   | nullable                               |
| `last_used_at`       | TIMESTAMP | nullable; stamped on each infer call   |
| `idle_timeout_hours` | FLOAT     | default 2.0                            |
| `created_at`         | TIMESTAMP |                                        |
| `updated_at`         | TIMESTAMP |                                        |

---

## API Contract

```
GET    /api/inferences              → list[InferenceInstance]
POST   /api/inferences              → InferenceInstance          body: InferenceConfig
GET    /api/inferences/{id}         → InferenceInstance
POST   /api/inferences/{id}/start   → InferenceInstance
POST   /api/inferences/{id}/stop    → InferenceInstance
DELETE /api/inferences/{id}         → 204 (only if unloaded|terminated)
```

### State machine

```
unloaded ──/start──► initialising ──(pod ready)──► ready
                                   └──(pod error)──► error
ready    ──/stop───► terminated
error    ──/stop───► terminated
any      ──(idle timeout)──► terminated
terminated ──/start──► initialising  (restart is allowed)
```

### OpenRouter shortcut

For `backend=openrouter`, `/start` skips K8s and immediately sets `status=ready`.

---

## Kubernetes Pod Design (TASK-13.4)

### `K8sPodAdapter` interface

```python
class K8sPodAdapter:
    def start(self, instance: InferenceInstance) -> str:
        """Launch pod; return pod_name."""

    def stop(self, pod_name: str) -> None:
        """Delete pod."""

    def pod_status(self, pod_name: str) -> Literal["pending", "running", "failed", "unknown"]:
        """Non-blocking poll of pod phase."""
```

Pod spec: single container using `ghcr.io/ggerganov/llama.cpp:server`, GGUF mounted
via an init-container that downloads from storage. `restartPolicy: Never`.

### Configuration env vars

| Var                            | Default          | Purpose                       |
|--------------------------------|------------------|-------------------------------|
| `KUBECONFIG`                   | `~/.kube/config` | K8s credentials               |
| `K8S_NAMESPACE`                | `default`        | Namespace for inference pods  |
| `INFERENCE_IDLE_TIMEOUT_HOURS` | `2`              | Idle shutdown threshold       |
| `K8S_MOCK`                     | `false`          | Use fake adapter (local dev)  |

---

## Idle Shutdown Design (TASK-13.5)

```python
async def idle_shutdown_loop(store: InferenceStorePort, k8s: K8sPodAdapter):
    while True:
        await asyncio.sleep(300)  # poll every 5 minutes
        timeout_hours = float(os.getenv("INFERENCE_IDLE_TIMEOUT_HOURS", "2"))
        cutoff = datetime.utcnow() - timedelta(hours=timeout_hours)
        for instance in store.list_by_status(InferenceStatus.READY):
            if instance.last_used_at and instance.last_used_at < cutoff:
                await stop_instance(instance, store, k8s)
```

Started in `app.lifespan`. `last_used_at` is stamped inside inference routes after
each successful infer call.

---

## UI Design (TASK-13.6)

### InferencePage layout

```
┌──────────────────────────────────────────────────────────────┐
│  Inference Instances                          [+ New Instance]│
├──────────────┬──────────────┬────────────┬────────────┬──────┤
│  Model       │  Backend     │  Status    │  Last used │ Act. │
├──────────────┼──────────────┼────────────┼────────────┼──────┤
│  SmolLM-v2   │  local       │ ● ready    │  2 min ago │ Stop │
│  Claude-haiku│  openrouter  │ ○ unloaded │  —         │Start │
│  SmolLM-v1   │  local       │ ✕ error    │  1 hr ago  │Start │
└──────────────┴──────────────┴────────────┴────────────┴──────┘
```

### Status badge colours

| Status        | Colour  |
|---------------|---------|
| `unloaded`    | grey    |
| `initialising`| yellow  |
| `ready`       | green   |
| `error`       | red     |
| `terminated`  | slate   |

### New files

| File | Purpose |
|------|---------|
| `ui/src/api/inferences.ts` | API client with typed functions |
| `ui/src/pages/InferencePage.tsx` | Main page: table + actions |
| `ui/src/components/InferenceStatusBadge.tsx` | Reusable status chip |

### Updated files

| File | Change |
|------|--------|
| `ui/src/types/index.ts` | Add `InferenceStatus`, `InferenceInstance` |
| `ui/src/App.tsx` | Add `/inferences` route |
| Main nav component | Add **Inference** tab |

---

## Task Dependencies & Build Order

```
TASK-13.1  (domain model + DB store)      ← start here
    │
    ├─► TASK-13.2  (REST API)             ┐
    ├─► TASK-13.3  (auto-create activity) │ run in parallel
    └─► TASK-13.4  (K8s adapter)          ┘
                │
                ▼
        TASK-13.5  (idle shutdown)   ← needs 13.2 + 13.4
        TASK-13.6  (UI)              ← needs 13.2 (API contract)
```

---

## Testing Strategy

| Task | Test file | What is tested |
|------|-----------|----------------|
| 13.1 | `tests/unit/test_inference_store.py` | CRUD, status transitions |
| 13.2 | `tests/integration/test_inferences_api.py` | All 6 endpoints, auth, state guards |
| 13.3 | `tests/unit/test_create_inference_activity.py` | Mock store, workflow hook |
| 13.4 | `tests/unit/test_k8s_adapter.py` | Mock k8s client, pod phases |
| 13.5 | `tests/unit/test_idle_shutdown.py` | Time travel via mock datetime |
| 13.6 | MSW handlers + RTL page tests | Table render, start/stop flows |

**Coverage gate: 80% minimum on all new files.**

---

## Open Questions / Risks

| # | Question | Risk | Mitigation |
|---|----------|------|------------|
| 1 | K8s cluster available in dev? | High | `K8S_MOCK=true` fake adapter |
| 2 | GGUF download latency before pod ready? | Medium | `initialising` status + polling UI |
| 3 | One pod per instance or shared? | Medium | One pod per instance; idle shutdown controls cost |
| 4 | Epic-12 dataset IDs in instances? | Low | `run_id` optional; dataset info from run record |
