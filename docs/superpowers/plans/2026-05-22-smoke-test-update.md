# Smoke Test Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `scripts/smoke_test.py` to test the full ML workflow — model creation, dataset upload, run trigger with polling, inference, and 5-table DB check — with cleanup on exit.

**Architecture:** The script is a single self-contained Python file. Helper functions are added for each new API operation. A `try/finally` block in `main()` ensures cleanup (DELETE model, DELETE dataset) always runs. Polling uses a simple `time.sleep` loop.

**Tech Stack:** Python 3.12+, `httpx`, `io`, `json`, `time`, `subprocess` — stdlib + httpx only (no new deps).

**Spec:** [docs/superpowers/specs/2026-05-22-smoke-test-update-design.md](../specs/2026-05-22-smoke-test-update-design.md)

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `scripts/smoke_test.py` | **Rewrite** | Full smoke test with new workflow |
| `docs/superpowers/specs/2026-05-22-smoke-test-update-design.md` | **New** | Design spec |
| `docs/superpowers/plans/2026-05-22-smoke-test-update.md` | **New** | This plan |

---

## Task 1: Add helper functions for new API operations

**Files:**
- Modify: `scripts/smoke_test.py`

Add four new helpers (`create_model`, `upload_dataset`, `trigger_run`, `poll_run_until_started`) and update the import block at the top of the file.

- [ ] **Step 1: Update the import block at the top of `scripts/smoke_test.py`**

Replace the existing imports (lines 1–10):

```python
#!/usr/bin/env python3
"""Post-deploy smoke test — validates the live API endpoints."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

import httpx
```

- [ ] **Step 2: Add helper functions between `check()` and `main()`**

Insert the following block immediately after the `check()` function and before `def main()`:

```python
def create_model(client: httpx.Client, api_url: str, headers: dict) -> dict:
    """POST /api/models — return created model record."""
    payload = {
        "name": "smoke-test-model",
        "description": "Created by smoke test — safe to delete",
        "base_model": "HuggingFaceTB/SmolLM2-360M",
        "remote_backend": "local",
        "skip_generate": True,
    }
    resp = client.post(f"{api_url}/api/models", json=payload, headers=headers)
    return check("POST /api/models", resp, expected_status=201)


def upload_dataset(client: httpx.Client, api_url: str, headers: dict) -> dict:
    """POST /api/datasets — upload a tiny synthetic JSONL; return dataset record."""
    lines = [
        json.dumps({"prompt": "scene tick=1 hunger=0.8", "completion": "EAT bowl1"}),
        json.dumps({"prompt": "scene tick=2 boredom=0.9", "completion": "PLAY toy1"}),
    ]
    content = "\n".join(lines).encode()
    resp = client.post(
        f"{api_url}/api/datasets",
        data={"name": "smoke-test-dataset", "dataset_type": "train", "description": "Smoke test"},
        files={"file": ("smoke.jsonl", io.BytesIO(content), "application/x-ndjson")},
        headers=headers,
    )
    return check("POST /api/datasets", resp, expected_status=201)


def trigger_run(
    client: httpx.Client,
    api_url: str,
    headers: dict,
    model_id: str,
    dataset_id: str,
) -> dict:
    """POST /api/runs/trigger — return {workflow_id, run_id}."""
    payload = {
        "model_id": model_id,
        "train_dataset_id": dataset_id,
        "skip_generate": True,
        "num_train_samples": 2,
        "num_eval_samples": 2,
    }
    resp = client.post(f"{api_url}/api/runs/trigger", json=payload, headers=headers)
    return check("POST /api/runs/trigger", resp, expected_status=202)


def poll_run_until_started(
    client: httpx.Client,
    api_url: str,
    headers: dict,
    run_id: str,
    timeout_seconds: int = 60,
    poll_interval: int = 5,
) -> dict:
    """Poll GET /api/runs/{run_id} until status moves past 'pending'. Return final run record."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        resp = client.get(f"{api_url}/api/runs/{run_id}", headers=headers)
        run = check(f"GET /api/runs/{run_id}", resp)
        status = run.get("status", "")
        if status != "pending":
            return run
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"ERROR: run {run_id} still 'pending' after {timeout_seconds}s",
                file=sys.stderr,
            )
            sys.exit(1)
        time.sleep(min(poll_interval, remaining))
```

- [ ] **Step 3: Verify the file compiles cleanly**

```bash
python -m py_compile scripts/smoke_test.py && echo "OK"
```

Expected:
```
OK
```

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "feat: add smoke test helpers — create_model, upload_dataset, trigger_run, poll_run_until_started"
```

---

## Task 2: Rewrite `main()` with the full 10-step workflow

**Files:**
- Modify: `scripts/smoke_test.py`

Replace the body of `def main() -> None:` with the new 10-step flow. The auth and health blocks are retained unchanged; steps 3–10 are new.

- [ ] **Step 1: Replace the entire `def main() -> None:` function**

```python
def main() -> None:
    print("=== Smoke Tests ===\n")

    api_url = require_env("API_URL").rstrip("/")
    auth0_domain = require_env("AUTH0_DOMAIN")
    auth0_client_id = require_env("AUTH0_MGMT_CLIENT_ID")
    auth0_client_secret = require_env("AUTH0_MGMT_CLIENT_SECRET")
    auth0_audience = require_env("AUTH0_AUDIENCE")

    client = httpx.Client(timeout=30)

    # 1. Authenticate via Auth0 M2M client credentials
    token_url = f"https://{auth0_domain}/oauth/token"
    client_id_hint = auth0_client_id[:6] + "..." if len(auth0_client_id) > 6 else auth0_client_id
    print("-- Authenticating via Auth0...")
    print(f"   token_url : {token_url}")
    print(f"   client_id : {client_id_hint}")
    print(f"   audience  : {auth0_audience}")
    token_resp = client.post(
        token_url,
        json={
            "grant_type": "client_credentials",
            "client_id": auth0_client_id,
            "client_secret": auth0_client_secret,
            "audience": auth0_audience,
        },
    )
    print(f"   status    : {token_resp.status_code}")
    if token_resp.status_code != 200:
        print(f"ERROR: Auth0 token exchange failed ({token_resp.status_code})", file=sys.stderr)
        print(f"   response  : {token_resp.text}", file=sys.stderr)
        sys.exit(1)
    access_token = token_resp.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    print("OK — token acquired\n")

    # 2. Health check (no auth required)
    health = check("GET /health", client.get(f"{api_url}/health"))
    print(f"OK — status={health.get('status')}\n")

    # Resources created during the test — cleaned up in the finally block below
    model_id: str | None = None
    dataset_id: str | None = None

    try:
        # 3. Create a model
        model = create_model(client, api_url, auth_headers)
        model_id = model["id"]
        print(f"OK — model_id={model_id}\n")

        # 4. Verify the model appears in the listing
        models = check("GET /api/models", client.get(f"{api_url}/api/models", headers=auth_headers))
        model_ids = [m["id"] for m in models]
        if model_id not in model_ids:
            print(
                f"ERROR: created model {model_id} not found in GET /api/models listing",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"OK — {len(models)} model(s) returned, created model present\n")

        # 5. Upload a training dataset
        dataset = upload_dataset(client, api_url, auth_headers)
        dataset_id = dataset["id"]
        print(f"OK — dataset_id={dataset_id}\n")

        # 6. Verify the dataset appears in the listing
        datasets = check(
            "GET /api/datasets",
            client.get(f"{api_url}/api/datasets", headers=auth_headers),
        )
        dataset_ids = [d["id"] for d in datasets]
        if dataset_id not in dataset_ids:
            print(
                f"ERROR: created dataset {dataset_id} not found in GET /api/datasets listing",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"OK — {len(datasets)} dataset(s) returned, created dataset present\n")

        # 7. Trigger a training run and poll until it moves past 'pending'
        run_trigger = trigger_run(client, api_url, auth_headers, model_id, dataset_id)
        run_id = run_trigger["run_id"]
        workflow_id = run_trigger["workflow_id"]
        print(f"OK — run_id={run_id} workflow_id={workflow_id}")
        print("   Polling until run status moves past 'pending'...")
        run = poll_run_until_started(client, api_url, auth_headers, run_id)
        print(f"OK — run status={run['status']}\n")

        # 8. Inference — minimal scene with a bowl so EAT is a valid candidate
        infer_payload = {
            "scene": {
                "objects": [{"id": "bowl1", "type": "bowl", "distance": 1.5}],
                "tick": 1,
            },
            "pet_stats": {
                "hunger": 0.8,
                "boredom": 0.3,
                "social": 0.2,
                "toilet": 0.1,
                "tiredness": 0.4,
            },
        }
        infer_resp = client.post(
            f"{api_url}/infer", json=infer_payload, headers=auth_headers, timeout=120
        )
        print("-- POST /infer...")
        if (
            infer_resp.status_code == 503
            and infer_resp.json().get("detail", {}).get("error") == "inference_disabled"
        ):
            print("OK — inference disabled (no model loaded)\n")
        else:
            infer = check("POST /infer", infer_resp)
            print(f"OK — action={infer['action']}\n")

        # 9. Database tables via kubectl
        print("-- Checking database tables...")
        result = subprocess.run(
            [
                "kubectl", "exec", "llm-api-db-0", "--",
                "psql", "-U", "aipet", "-d", "aipet", "-t", "-c",
                "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: kubectl exec failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        tables = result.stdout
        print(tables)
        for table in (
            "alembic_version",
            "datasets",
            "inference_instances",
            "training_models",
            "training_runs",
        ):
            if table not in tables:
                print(f"ERROR: expected table '{table}' not found in database", file=sys.stderr)
                sys.exit(1)
        print("OK — all required tables present\n")

    finally:
        # 10. Cleanup — always runs, even on test failure
        if dataset_id:
            print(f"-- Cleanup: DELETE /api/datasets/{dataset_id}...")
            del_resp = client.delete(
                f"{api_url}/api/datasets/{dataset_id}", headers=auth_headers
            )
            if del_resp.status_code == 204:
                print("OK — dataset deleted\n")
            else:
                print(
                    f"WARN: dataset delete returned {del_resp.status_code} — manual cleanup may be needed",
                    file=sys.stderr,
                )

        if model_id:
            print(f"-- Cleanup: DELETE /api/models/{model_id}...")
            del_resp = client.delete(
                f"{api_url}/api/models/{model_id}", headers=auth_headers
            )
            if del_resp.status_code == 204:
                print("OK — model deleted\n")
            else:
                print(
                    f"WARN: model delete returned {del_resp.status_code} — manual cleanup may be needed",
                    file=sys.stderr,
                )

    print("=== Smoke tests passed ===")
```

- [ ] **Step 2: Verify the full file compiles**

```bash
python -m py_compile scripts/smoke_test.py && echo "OK"
```

Expected:
```
OK
```

- [ ] **Step 3: Confirm missing `API_URL` exits cleanly (no live API needed)**

```bash
python scripts/smoke_test.py 2>&1 | head -3
```

Expected:
```
ERROR: API_URL environment variable is required
```

- [ ] **Step 4: Count function definitions as a sanity check**

```bash
grep -c "^def " scripts/smoke_test.py
```

Expected:
```
7
```

(Functions: `require_env`, `check`, `create_model`, `upload_dataset`, `trigger_run`, `poll_run_until_started`, `main`)

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "feat: rewrite smoke test main() — full ML workflow with model, dataset, run, inference, DB check, cleanup"
```

---

## Task 3: Commit design docs

**Files:**
- Add: `docs/superpowers/specs/2026-05-22-smoke-test-update-design.md`
- Add: `docs/superpowers/plans/2026-05-22-smoke-test-update.md`

- [ ] **Step 1: Commit the spec and plan**

```bash
git add docs/superpowers/specs/2026-05-22-smoke-test-update-design.md
git add docs/superpowers/plans/2026-05-22-smoke-test-update.md
git commit -m "docs: add smoke test update spec and implementation plan"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 10 steps from the spec flow are implemented in Task 2's `main()`
- [x] **Placeholder scan:** No TBD/TODO/placeholder — all code is complete and runnable
- [x] **Type consistency:** `create_model` returns `dict` → `model["id"]`; `upload_dataset` returns `dict` → `dataset["id"]`; `trigger_run` returns `dict` → `run_trigger["run_id"]` — all match FastAPI route `response_model` schemas
- [x] **Cleanup guard:** `try/finally` wraps steps 3–9; cleanup always runs
- [x] **DB table list:** 5 tables (`alembic_version`, `datasets`, `inference_instances`, `training_models`, `training_runs`) match `__tablename__` values in `src/adapters/database/*.py`
- [x] **Import block:** `io`, `json`, `time` added to existing stdlib import block — no new third-party deps
- [x] **httpx multipart:** `upload_dataset` uses `data=` for form fields and `files=` for the file — correct httpx multipart API
