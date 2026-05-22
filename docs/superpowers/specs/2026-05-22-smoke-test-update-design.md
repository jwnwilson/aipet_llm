# Smoke Test Update — Design Spec

> Created: 2026-05-22

## Problem

The existing `scripts/smoke_test.py` was written before several major API additions:
- `POST /api/models` (create model)
- `POST /api/datasets` (upload dataset, multipart)
- `POST /api/runs/trigger` (trigger training run)
- `GET /api/runs/{id}` (poll run status)
- New DB tables: `datasets`, `inference_instances`

The smoke test is missing end-to-end coverage of the full ML workflow.

## Goal

Update `scripts/smoke_test.py` to test the complete lifecycle:
1. Create a model
2. Upload a training dataset
3. Trigger a training run and poll until it moves past `pending`
4. Test inference on the existing activated GGUF (or gracefully skip if disabled)
5. Verify all 5 DB tables exist
6. Clean up created resources (model, dataset) regardless of test outcome

## Decisions

- **Inference backend:** Use existing activated GGUF via `POST /infer`. Skip gracefully on `inference_disabled`.
- **Run trigger:** Fire `POST /api/runs/trigger`, then poll `GET /api/runs/{id}` every 5 s up to 60 s until `status != "pending"`.
- **Cleanup:** `try/finally` wraps the test body so DELETE calls run even on failure.
- **Dataset:** Generate a tiny 2-line synthetic JSONL in-memory (no on-disk file needed).

## Test Flow

```
1. Auth0 M2M token exchange
2. GET /health
3. POST /api/models            → model_id
4. GET /api/models             → model appears in list
5. POST /api/datasets          → dataset_id  (multipart, synthetic JSONL)
6. GET /api/datasets           → dataset appears in list
7. POST /api/runs/trigger      → run_id + workflow_id
   Poll GET /api/runs/{run_id} until status != "pending" (60 s max)
8. POST /infer                 → action returned (or inference_disabled OK)
9. kubectl DB check            → 5 tables present
10. Cleanup: DELETE /api/models/{id}, DELETE /api/datasets/{id}
```

## DB Tables Expected

```
alembic_version, datasets, inference_instances, training_models, training_runs
```

## Environment Variables Required

```
API_URL, AUTH0_DOMAIN, AUTH0_MGMT_CLIENT_ID, AUTH0_MGMT_CLIENT_SECRET, AUTH0_AUDIENCE
```
