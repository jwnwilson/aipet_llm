#!/usr/bin/env python3
"""Post-deploy smoke test — validates the live API endpoints.

Full pipeline exercised (k8s backend):
  dataset upload → k8s training Job → eval → inference pod → /infer → cleanup

Set REMOTE_BACKEND env var to switch backends (default: k8s).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

import httpx


def require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        print(f"ERROR: {name} environment variable is required", file=sys.stderr)
        sys.exit(1)
    return val


def check(label: str, resp: httpx.Response, expected_status: int = 200) -> dict:
    print(f"-- {label}...")
    if resp.status_code != expected_status:
        print(f"ERROR: expected HTTP {expected_status}, got {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)
    return resp.json()


def create_model(
    client: httpx.Client,
    api_url: str,
    headers: dict[str, str],
    remote_backend: str = "k8s",
) -> dict:
    """POST /api/models — return created model record."""
    payload = {
        "name": "smoke-test-model",
        "description": "Created by smoke test — safe to delete",
        "base_model": "HuggingFaceTB/SmolLM2-360M",
        "remote_backend": remote_backend,
        "skip_generate": True,
    }
    resp = client.post(f"{api_url}/api/models", json=payload, headers=headers)
    return check("POST /api/models", resp, expected_status=201)


def upload_dataset(client: httpx.Client, api_url: str, headers: dict[str, str]) -> dict:
    """POST /api/datasets — upload a tiny synthetic JSONL; return dataset record."""
    lines = [
        json.dumps({"prompt": "scene tick=1 hunger=0.8", "completion": "EAT bowl1"}),
        json.dumps({"prompt": "scene tick=2 boredom=0.9", "completion": "PLAY toy1"}),
    ]
    content = ("\n".join(lines) + "\n").encode()
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
    headers: dict[str, str],
    model_id: str,
    dataset_id: str,
    remote_backend: str = "k8s",
) -> dict:
    """POST /api/runs/trigger — return {workflow_id, run_id}."""
    payload = {
        "model_id": model_id,
        "train_dataset_id": dataset_id,
        "remote_backend": remote_backend,
        "skip_generate": True,
        "num_train_samples": 2,
        "num_eval_samples": 2,
    }
    resp = client.post(f"{api_url}/api/runs/trigger", json=payload, headers=headers)
    return check("POST /api/runs/trigger", resp, expected_status=202)


_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def poll_run_to_completion(
    client: httpx.Client,
    api_url: str,
    headers: dict[str, str],
    run_id: str,
    timeout_seconds: int = 900,
    poll_interval: int = 10,
) -> dict:
    """Poll GET /api/runs/{run_id} until status reaches a terminal state.

    Exits 1 if the run fails, is cancelled, or times out.
    Returns the final run record on success (status == 'completed').
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        resp = client.get(f"{api_url}/api/runs/{run_id}", headers=headers)
        run = check(f"GET /api/runs/{run_id}", resp)
        status = run.get("status", "")
        if status in _TERMINAL_STATUSES:
            if status != "completed":
                detail = run.get("progress_detail") or ""
                print(
                    f"ERROR: run {run_id} ended with status='{status}'"
                    f"{': ' + detail if detail else ''}",
                    file=sys.stderr,
                )
                sys.exit(1)
            return run
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"ERROR: run {run_id} did not complete within {timeout_seconds}s"
                f" (current status='{status}')",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"   run status={status} — waiting...")
        time.sleep(min(poll_interval, remaining))


def start_inference_pod(
    client: httpx.Client,
    api_url: str,
    headers: dict[str, str],
    model_id: str,
) -> dict:
    """Find or create an inference instance for model_id, then POST /start.

    Returns the inference instance record after starting.
    """
    # Find existing instance for this model
    instances_resp = client.get(f"{api_url}/api/inferences", headers=headers)
    instances = check("GET /api/inferences", instances_resp)

    instance = next((i for i in instances if i.get("model_id") == model_id), None)

    if instance is None:
        # Create a new inference instance
        create_resp = client.post(
            f"{api_url}/api/inferences",
            json={"model_id": model_id},
            headers=headers,
        )
        instance = check("POST /api/inferences", create_resp, expected_status=201)

    instance_id = instance["id"]
    # Start the pod
    start_resp = client.post(
        f"{api_url}/api/inferences/{instance_id}/start",
        headers=headers,
    )
    return check(f"POST /api/inferences/{instance_id}/start", start_resp)


def poll_inference_available(
    client: httpx.Client,
    api_url: str,
    headers: dict[str, str],
    instance_id: str,
    timeout_seconds: int = 300,
    poll_interval: int = 10,
) -> dict:
    """Poll GET /api/inferences/{id} until status == 'available'.

    Exits 1 on 'failed' status or timeout.
    Returns the instance record when available.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        resp = client.get(f"{api_url}/api/inferences/{instance_id}", headers=headers)
        instance = check(f"GET /api/inferences/{instance_id}", resp)
        status = instance.get("status", "")
        if status == "available":
            return instance
        if status == "failed":
            print(
                f"ERROR: inference instance {instance_id} entered 'failed' state",
                file=sys.stderr,
            )
            sys.exit(1)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"ERROR: inference instance {instance_id} not available after {timeout_seconds}s"
                f" (current status='{status}')",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"   inference status={status} — waiting...")
        time.sleep(min(poll_interval, remaining))


def main() -> None:
    print("=== Smoke Tests ===\n")

    api_url = require_env("API_URL").rstrip("/")
    auth0_domain = require_env("AUTH0_DOMAIN")
    auth0_client_id = require_env("AUTH0_MGMT_CLIENT_ID")
    auth0_client_secret = require_env("AUTH0_MGMT_CLIENT_SECRET")
    auth0_audience = require_env("AUTH0_AUDIENCE")
    remote_backend = os.environ.get("REMOTE_BACKEND", "k8s")

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
        try:
            err_desc = token_resp.json().get("error_description", token_resp.json().get("error", "unknown"))
        except Exception:
            err_desc = "could not parse error response"
        print(f"   error     : {err_desc}", file=sys.stderr)
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
    run_id: str | None = None
    instance_id: str | None = None

    try:
        # 3. Create a model (with remote_backend wired in)
        model = create_model(client, api_url, auth_headers, remote_backend=remote_backend)
        model_id = model["id"]
        print(f"OK — model_id={model_id} remote_backend={remote_backend}\n")

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

        # 7. Trigger a training run and wait for completion
        run_trigger = trigger_run(
            client, api_url, auth_headers, model_id, dataset_id,
            remote_backend=remote_backend,
        )
        run_id = run_trigger["run_id"]
        workflow_id = run_trigger["workflow_id"]
        print(f"OK — run_id={run_id} workflow_id={workflow_id}")
        print("   Polling until run completes...")
        run = poll_run_to_completion(client, api_url, auth_headers, run_id)
        print(f"OK — run status={run.get('status')}\n")

        # 8. Start an inference pod for the trained model
        print("   Starting inference pod for trained model...")
        instance = start_inference_pod(client, api_url, auth_headers, model_id)
        instance_id = instance["id"]
        print(f"OK — inference instance started: instance_id={instance_id}\n")

        # 9. Wait for the inference pod to become available
        print("   Waiting for inference pod to become available...")
        poll_inference_available(client, api_url, auth_headers, instance_id)
        print("OK — inference pod available\n")

        # 10. Run inference via the pod
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
        print(f"-- POST /api/inferences/{instance_id}/infer...")
        infer_resp = client.post(
            f"{api_url}/api/inferences/{instance_id}/infer",
            json=infer_payload,
            headers=auth_headers,
            timeout=120,
        )
        infer = check(f"POST /api/inferences/{instance_id}/infer", infer_resp)
        action = infer.get("action")
        if not action:
            print("ERROR: inference response missing 'action' field", file=sys.stderr)
            sys.exit(1)
        print(f"OK — action={action}\n")

        # 11. Database tables via kubectl
        db_pod = os.environ.get("DB_POD_NAME", "llm-api-db-0")
        db_ns  = os.environ.get("DB_NAMESPACE", "default")
        db_user = os.environ.get("DB_USER", "aipet")
        db_name = os.environ.get("DB_NAME", "aipet")
        print(f"-- Checking database tables (pod={db_pod}, ns={db_ns})...")
        result = subprocess.run(
            [
                "kubectl", "exec", db_pod,
                "--namespace", db_ns,
                "--",
                "psql", "-U", db_user, "-d", db_name, "-t", "-c",
                "SELECT tablename FROM pg_tables WHERE schemaname=\'public\' ORDER BY tablename;",
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
        # 12. Cleanup — always runs, even on test failure
        # Stop and delete inference instance first (it holds k8s resources)
        if instance_id:
            print(f"-- Cleanup: stop inference instance {instance_id}...")
            try:
                stop_resp = client.post(
                    f"{api_url}/api/inferences/{instance_id}/stop",
                    headers=auth_headers,
                )
                if stop_resp.status_code in (200, 204):
                    print("OK — inference instance stopped")
                else:
                    print(
                        f"WARN: stop returned {stop_resp.status_code} — continuing cleanup",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"WARN: inference stop raised {exc} — continuing cleanup", file=sys.stderr)

            print(f"-- Cleanup: DELETE /api/inferences/{instance_id}...")
            try:
                del_resp = client.delete(
                    f"{api_url}/api/inferences/{instance_id}", headers=auth_headers
                )
                if del_resp.status_code == 204:
                    print("OK — inference instance deleted\n")
                else:
                    print(
                        f"WARN: inference delete returned {del_resp.status_code}"
                        " — manual cleanup may be needed",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(
                    f"WARN: inference cleanup raised {exc} — manual cleanup may be needed",
                    file=sys.stderr,
                )

        if dataset_id:
            print(f"-- Cleanup: DELETE /api/datasets/{dataset_id}...")
            try:
                del_resp = client.delete(
                    f"{api_url}/api/datasets/{dataset_id}", headers=auth_headers
                )
                if del_resp.status_code == 204:
                    print("OK — dataset deleted\n")
                else:
                    print(
                        f"WARN: dataset delete returned {del_resp.status_code}"
                        " — manual cleanup may be needed",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(
                    f"WARN: dataset cleanup raised {exc} — manual cleanup may be needed",
                    file=sys.stderr,
                )

        if run_id:
            print(f"-- Cleanup: DELETE /api/runs/{run_id}...")
            try:
                del_resp = client.delete(
                    f"{api_url}/api/runs/{run_id}", headers=auth_headers
                )
                if del_resp.status_code == 204:
                    print("OK — run deleted\n")
                else:
                    print(
                        f"WARN: run delete returned {del_resp.status_code}"
                        " — manual cleanup may be needed",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"WARN: run cleanup raised {exc} — manual cleanup may be needed", file=sys.stderr)

        if model_id:
            print(f"-- Cleanup: DELETE /api/models/{model_id}...")
            try:
                del_resp = client.delete(
                    f"{api_url}/api/models/{model_id}", headers=auth_headers
                )
                if del_resp.status_code == 204:
                    print("OK — model deleted\n")
                else:
                    print(
                        f"WARN: model delete returned {del_resp.status_code}"
                        " — manual cleanup may be needed",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"WARN: model cleanup raised {exc} — manual cleanup may be needed", file=sys.stderr)

    print("=== Smoke tests passed ===")


if __name__ == "__main__":
    main()
