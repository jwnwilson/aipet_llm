# Run Failure & Cancellation Handling — Design

**Date:** 2026-05-20  
**Status:** Approved

---

## Problem

Training runs can fail mid-pipeline (activity exception) or be cancelled by the user via Temporal. Neither case currently updates the run record — the run is left permanently stuck in an intermediate status (`TRAINING`, `EVALUATING`, etc.) with no indication of what went wrong.

---

## Data Flow

```
TrainingPipelineWorkflow.run(config: ExperimentConfig)
│
├── try:
│   │
│   ├── update_run_status_activity(run_id, "generating")
│   ├── generate_dataset_activity(DatasetConfig)
│   │       └── on error → raises ApplicationError("generate_dataset failed: …")
│   │
│   ├── update_run_status_activity(run_id, "training")
│   ├── train_activity(TrainConfig)
│   │       └── on error → raises ApplicationError("train failed: …")
│   │
│   ├── update_run_status_activity(run_id, "evaluating")
│   ├── evaluate_activity(EvalConfig)  → EvalResult(valid_pct, passed)
│   │       └── on error → raises ApplicationError("evaluate failed: …")
│   │
│   └── finalise_run_activity(run_id, passed, valid_pct)
│           └── sets COMPLETED (passed=True) or FAILED (passed=False, threshold miss)
│
├── except ApplicationError as exc:
│   │   ← any activity threw (generate / train / evaluate / export / finalise)
│   └── fail_run_activity(run_id, reason=str(exc), status_value="failed")
│           └── store.fail_run(run_id, reason, RunStatus.FAILED)
│               sets status="failed", progress_detail=reason
│
└── except CancelledError:
    │   ← Temporal user-initiated cancel (workflow.cancel() called externally)
    ├── [CancellationScope.detached()]
    │   └── fail_run_activity(run_id, reason="cancelled by user", status_value="cancelled")
    │           └── store.fail_run(run_id, reason, RunStatus.CANCELLED)
    │               sets status="cancelled", progress_detail="cancelled by user"
    └── raise   ← must re-raise so Temporal registers the cancellation
```

---

## Changes

### 1. Domain — `src/domain/models.py`

Add `CANCELLED` to `RunStatus`:

```python
class RunStatus(str, Enum):
    ...
    CANCELLED = "cancelled"   # NEW — user-initiated Temporal cancel
```

### 2. Domain port — `src/domain/ports.py`

Add `fail_run` to `RunStorePort`:

```python
@abstractmethod
def fail_run(
    self,
    run_id: str,
    reason: str,
    status: RunStatus = RunStatus.FAILED,
) -> RunRecord | None:
    """Mark a run as failed or cancelled, persisting the reason in progress_detail.
    Returns updated record, or None if run_id not found."""
```

### 3. Database adapter — `src/adapters/database/run_store.py`

Implement `fail_run` on `SQLAlchemyRunStore` — single session, two field updates:

```python
def fail_run(self, run_id, reason, status=RunStatus.FAILED):
    with Session(self._engine) as db:
        row = db.get(_RunRow, run_id)
        if row is None:
            return None
        row.status = status.value
        row.progress_detail = reason
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(row)
        return _row_to_domain(row)
```

No migration required — `status` is `String(32)`, no DB-level enum constraint.

### 4. Temporal activity — `src/interactors/temporal/activities.py`

New activity, placed alongside `update_run_status_activity`:

```python
@activity.defn
async def fail_run_activity(run_id: str, reason: str, status_value: str = "failed") -> None:
    """Mark a run as failed or cancelled with a reason string."""
    store = _get_run_store()
    store.fail_run(run_id, reason, RunStatus(status_value))
```

### 5. Temporal workflow — `src/interactors/temporal/workflows.py`

Wrap the body of `TrainingPipelineWorkflow.run()` in try/except. Register `fail_run_activity` in the worker.

```python
async def run(self, config: ExperimentConfig) -> None:
    run_id = config.run_id
    try:
        # … existing stage logic unchanged …
    except asyncio.CancelledError:
        with workflow.CancellationScope.detached():
            await workflow.execute_activity(
                fail_run_activity,
                args=[run_id, "cancelled by user", "cancelled"],
                start_to_close_timeout=timedelta(seconds=30),
            )
        raise
    except Exception as exc:
        await workflow.execute_activity(
            fail_run_activity,
            args=[run_id, str(exc), "failed"],
            start_to_close_timeout=timedelta(seconds=30),
        )
```

### 6. UI — `ui/src/types/index.ts`

```typescript
export type RunStatus =
  | 'pending' | 'generating' | 'training' | 'evaluating'
  | 'exporting' | 'running' | 'completed' | 'failed'
  | 'cancelled'   // NEW
```

### 7. UI — `ui/src/components/RunStatusBadge.tsx`

```typescript
cancelled: { label: 'Cancelled', className: 'bg-yellow-100 text-yellow-800' },
```

---

## Testing

- Unit: `test_temporal_workflow.py` — add test for activity exception → FAILED, and CancelledError → CANCELLED
- Unit: `test_temporal_activities.py` — add tests for `fail_run_activity`
- Unit: `test_run_store.py` (or equivalent) — add test for `fail_run` method
- Run full suite: `uv run python -m pytest tests/unit/ tests/integration/ -q`
