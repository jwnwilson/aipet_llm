# PR Review: #77 — fix: standardise model S3 paths to prevent 404 errors on load

**Reviewed**: 2026-06-01
**Author**: jwnwilson
**Branch**: worktree-fix-consistent-model-paths → main
**Decision**: REQUEST CHANGES

## Summary

The PR's core approach is correct — centralising path construction in `paths.py` is the right fix. However, `activate_run` still 404s because `download_model` is called with a key that has no `.gz` suffix, while `upload_model` always stores the object with `.gz`. The path prefix changed correctly; the compression suffix mismatch was not addressed.

---

## Findings

### HIGH

**[H1] `activate_run` still produces a 404 — `.gz` mismatch not fixed**
File: `src/interactors/api/routes/runs.py:476–479`

`upload_model` always appends `.gz` to the S3 key before uploading. The `save_gguf_path_activity` therefore stores e.g. `workflow/{id}/model/model.gguf.gz` in `TrainingModel.gguf_path`. But `activate_run` calls:
```python
gguf_key = workflow_model_key(run_id)          # → "workflow/{id}/model/model.gguf"
download_model(storage, gguf_key, local_path)   # no .gz → storage.download(key, dest)
```
`download_model` only decompresses when the key ends in `.gz`. With a bare `.gguf` key it passes through directly to `storage.download`, which will 404 because the actual S3 object ends in `.gguf.gz`.

`activate_model` avoids this by reading `model.gguf_path` from the DB (which already has `.gz`). `activate_run` should do the same — look up the model via `run.model_id` and use its stored `gguf_path`:

```python
def activate_run(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    model_store: ModelStorePort = Depends(get_model_store),  # add
    user: UserContext = Depends(require_approved),
) -> RunRecord:
    ...
    model = model_store.get(run.model_id)
    if model and model.gguf_path:
        gguf_key = model.gguf_path          # already includes .gz
    else:
        gguf_key = workflow_model_key(run_id)  # fallback for runs before save_gguf_path ran
    local_path = Path(f"data/workflow/{run_id}/model.gguf")
    download_model(storage, gguf_key, local_path)
```

---

### MEDIUM

**[M1] Path helper imports buried inside function bodies**
Files: `src/interactors/temporal/activities.py:659`, `src/interactors/api/routes/runs.py:467`

```python
# activities.py — inside export_activity
from adapters.storage.paths import standalone_model_key, workflow_model_key
```
Module-level imports are the Python convention and make dependencies visible at a glance. Move both imports to the top of their respective files.

---

**[M2] Stale path examples in `domain/models.py`**
File: `src/domain/models.py:120`

```python
# Example: "workflow/{run_id}/model.gguf" or "gguf/{model_name}.gguf"
```
Should read: `"workflow/{run_id}/model/{model_name}.gguf"` or `"model/{model_id}/{model_name}.gguf"`

---

**[M3] Stale path examples in `domain/ports.py`**
File: `src/domain/ports.py:34`

```
Keys are relative paths such as ``workflow/{run_id}/model.gguf``.
```
Update to the new canonical format.

---

**[M4] No direct unit tests for `paths.py` helpers**
File: `src/adapters/storage/paths.py`

The helpers are covered indirectly via `test_temporal_activities.py`, but a small focused test for edge cases (empty `model_id` → `model//model.gguf`) would catch silent bad keys before they reach S3.

---

### LOW

**[L1] Stale examples in storage adapter docstrings**
- `src/adapters/storage/local.py:14` — example still says `gguf/{model_id}.gguf`
- `src/adapters/storage/s3.py:16` — example still says `workflow/{run_id}/model.gguf`
- `src/interactors/cli/training/k8s_export.py:11` — env-var comment still cites old format

These are documentation only; no functional impact.

---

**[L2] No validation for empty inputs in path helpers**
File: `src/adapters/storage/paths.py`

`standalone_model_key("", "model")` silently produces `"model//model.gguf"` — a valid string that will 404 on S3 with no obvious cause. A one-line guard would make failures loud:
```python
if not model_id:
    raise ValueError("model_id must not be empty")
```

---

## Validation Results

| Check | Result |
|---|---|
| Tests (`pytest tests/unit/`) | 579 passed, 1 skipped |
| Lint / type-check | Skipped (no ruff/mypy config detected) |
| Build | N/A (Python, no build step) |

---

## Files Reviewed

| File | Change |
|---|---|
| `src/adapters/storage/paths.py` | Added |
| `src/adapters/storage/__init__.py` | Modified — re-exports helpers |
| `src/interactors/temporal/activities.py` | Modified — uses helpers in `export_activity` |
| `src/interactors/api/routes/runs.py` | Modified — uses helper in `activate_run` |
| `tests/unit/test_temporal_activities.py` | Modified — updated assertions |
| `CLAUDE.md` | Modified — S3 path table updated |

Also surveyed (not in diff, contain related references):
- `src/domain/models.py` — stale comment [M2]
- `src/domain/ports.py` — stale comment [M3]
- `src/adapters/storage/local.py`, `s3.py` — stale examples [L1]
- `src/interactors/cli/training/k8s_export.py` — stale comment [L1]