# Run Logs UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display Temporal workflow log output on the Run Detail page so users can debug training problems — and fix the `ERR_CONNECTION_REFUSED` error that breaks the deployed site.

**Architecture:** The `feature-log-streaming` worktree already contains all the pieces; this plan ports them into `abstract-domain-cli`. The backend adds two endpoints (`GET /logs`, `GET /logs/stream`) to `runs.py`. The frontend wires the existing `RunLogViewer` component into `RunDetailPage` using a new `useLogStream` hook that opens a Fetch-based SSE connection with Bearer auth support.

**Tech Stack:** FastAPI `StreamingResponse` (SSE, no extra deps), React Query (REST fallback for completed runs), Fetch API ReadableStream (live tail), TypeScript

**All paths are relative to** `.claude/worktrees/abstract-domain-cli/`

---

## Root-cause summary

| Issue | Cause | Fix |
|---|---|---|
| `ERR_CONNECTION_REFUSED` on deployed site | `client.ts` falls back to `http://localhost:8000` when `VITE_API_URL` is unset | Change fallback to `''` (relative URLs) |
| No logs in UI | `RunLogViewer` exists but is never rendered; no hook or API call fetches logs | Wire viewer into `RunDetailPage` |
| Missing backend endpoints | `runs.py` is missing `GET /logs` and `GET /logs/stream` | Port endpoints from `feature-log-streaming` |

---

## File map

| File | Action | Purpose |
|---|---|---|
| `ui/src/api/client.ts` | Modify | Fix baseURL fallback; export `getAuthToken()` |
| `ui/src/types/index.ts` | Modify | Add `RunLogsResponse` type |
| `ui/src/api/runs.ts` | Modify | Add `getRunLogs()` |
| `ui/src/hooks/useLogStream.ts` | **Create** | Fetch-based SSE hook with Bearer auth |
| `ui/src/pages/RunDetailPage.tsx` | Modify | Wire `RunLogViewer` + log data |
| `src/interactors/api/routes/runs.py` | Modify | Add log endpoints |
| `tests/integration/test_log_stream_api.py` | **Create** | Integration tests for new endpoints |

---

### Task 1 — Fix API base URL

**Files:**
- Modify: `ui/src/api/client.ts`

- [ ] **Step 1: Change the baseURL fallback**

  In `ui/src/api/client.ts`, change line 15:
  ```typescript
  // before
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
  // after
  baseURL: import.meta.env.VITE_API_URL ?? '',
  ```
  With an empty string, axios uses relative paths (`/api/runs/...`). In dev, the Vite proxy in `vite.config.ts` routes `/api → http://localhost:8000`. In production, nginx should proxy `/api` to the backend, or set `VITE_API_URL` to the deployed API URL (e.g. `https://api.llm.jwnwilson.co.uk`).

- [ ] **Step 2: Export `getAuthToken()` helper**

  Add this after the `setTokenGetter` function in `client.ts`:
  ```typescript
  export async function getAuthToken(): Promise<string | null> {
    if (!_tokenGetter) return null
    try {
      return await _tokenGetter()
    } catch {
      return null
    }
  }
  ```

- [ ] **Step 3: Verify Vite proxy is still present**

  Run: `grep -n proxy ui/vite.config.ts`

  Expected output contains: `'/api': 'http://localhost:8000'`

  No change needed — relative URLs in dev will hit Vite proxy.

- [ ] **Step 4: Commit**

  ```bash
  git add ui/src/api/client.ts
  git commit -m "fix: use relative API base URL so deployed site routes correctly"
  ```

---

### Task 2 — Add `RunLogsResponse` type

**Files:**
- Modify: `ui/src/types/index.ts`

- [ ] **Step 1: Add the type**

  After the `RunRecord` interface in `ui/src/types/index.ts`, add:
  ```typescript
  export interface RunLogsResponse {
    logs: string | null
    source: string | null
  }
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add ui/src/types/index.ts
  git commit -m "feat: add RunLogsResponse type"
  ```

---

### Task 3 — Add `getRunLogs()` API function

**Files:**
- Modify: `ui/src/api/runs.ts`

- [ ] **Step 1: Update the type import**

  In `ui/src/api/runs.ts`, update the first import to include `RunLogsResponse`:
  ```typescript
  import type { EvaluationData, RunLogsResponse, RunRecord, RunStatus } from '@/types'
  ```

- [ ] **Step 2: Add the function**

  Append to the end of `ui/src/api/runs.ts`:
  ```typescript
  export async function getRunLogs(id: string): Promise<RunLogsResponse> {
    const { data } = await apiClient.get<RunLogsResponse>(`/api/runs/${id}/logs`)
    return data
  }
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add ui/src/api/runs.ts ui/src/types/index.ts
  git commit -m "feat: add getRunLogs API function"
  ```

---

### Task 4 — Port log streaming backend endpoints

**Files:**
- Modify: `src/interactors/api/routes/runs.py`
- Create: `tests/integration/test_log_stream_api.py`

- [ ] **Step 1: Write failing integration tests**

  Create `tests/integration/test_log_stream_api.py`:
  ```python
  """Integration tests for run log endpoints."""
  import pytest
  from httpx import AsyncClient


  @pytest.mark.asyncio
  async def test_get_run_logs_404_for_unknown_run(client: AsyncClient) -> None:
      resp = await client.get("/api/runs/00000000-0000-0000-0000-000000000000/logs")
      assert resp.status_code == 404


  @pytest.mark.asyncio
  async def test_log_stream_404_for_unknown_run(client: AsyncClient) -> None:
      resp = await client.get("/api/runs/00000000-0000-0000-0000-000000000000/logs/stream")
      assert resp.status_code == 404


  @pytest.mark.asyncio
  async def test_get_run_logs_returns_null_when_no_log_file(
      client: AsyncClient, created_run_id: str
  ) -> None:
      resp = await client.get(f"/api/runs/{created_run_id}/logs")
      assert resp.status_code == 200
      data = resp.json()
      assert data["logs"] is None
      assert data["source"] is None


  @pytest.mark.asyncio
  async def test_log_stream_returns_text_event_stream(
      client: AsyncClient, created_run_id: str
  ) -> None:
      import pathlib
      log_file = pathlib.Path(f"data/workflow/{created_run_id}/logs.txt")
      log_file.parent.mkdir(parents=True, exist_ok=True)
      log_file.write_text("hello\nworld\n")

      resp = await client.get(
          f"/api/runs/{created_run_id}/logs/stream",
          headers={"Accept": "text/event-stream"},
      )
      assert resp.status_code == 200
      assert "text/event-stream" in resp.headers["content-type"]
      log_file.unlink(missing_ok=True)
  ```
  Note: `client` and `created_run_id` are fixtures from `tests/integration/conftest.py`. If `created_run_id` doesn't exist, check `conftest.py` for the correct fixture name for a run owned by the test user.

- [ ] **Step 2: Run to confirm FAIL**

  ```bash
  uv run pytest tests/integration/test_log_stream_api.py -v 2>&1 | head -40
  ```
  Expected: tests fail because the endpoints don't exist yet.

- [ ] **Step 3: Add imports to `runs.py`**

  In `src/interactors/api/routes/runs.py`, change the import block:
  ```python
  # before
  import logging
  import os
  import uuid
  from pathlib import Path

  from fastapi import APIRouter, Depends, HTTPException
  from pydantic import BaseModel

  # after
  import asyncio
  import logging
  import os
  import re
  import uuid
  from pathlib import Path

  from fastapi import APIRouter, Depends, HTTPException
  from fastapi.responses import StreamingResponse
  from pydantic import BaseModel
  ```

- [ ] **Step 4: Add `_UUID_HEX_RE` and `_log_path_for_run()` after `log = logging.getLogger(__name__)`**

  ```python
  _UUID_HEX_RE = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$")


  def _log_path_for_run(run_id: str) -> Path:
      if not _UUID_HEX_RE.match(run_id):
          raise HTTPException(status_code=404, detail="Run not found")
      return Path(f"data/workflow/{run_id}/logs.txt")
  ```

- [ ] **Step 5: Add `RunLogsResponse` schema**

  After the existing request schemas (after `ExportRequest`), add:
  ```python
  class RunLogsResponse(BaseModel):
      logs: str | None
      source: str | None
  ```

- [ ] **Step 6: Add `_TERMINAL_STATUSES` constant**

  After the existing `_CANCELLABLE_STATUSES` set (or near the top of the module with other constants), add:
  ```python
  _TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})
  ```

- [ ] **Step 7: Add `get_run_logs` endpoint**

  After the existing `GET /{run_id}/evaluation` endpoint, add:
  ```python
  @router.get("/{run_id}/logs", response_model=RunLogsResponse)
  def get_run_logs(
      run_id: str,
      run_store: RunStorePort = Depends(get_run_store),
      user: UserContext = Depends(require_approved),
  ) -> RunLogsResponse:
      run = run_store.get(run_id)
      if run is None:
          raise HTTPException(status_code=404, detail="Run not found")
      if run.owner_id is not None and run.owner_id != user.user_id:
          raise HTTPException(status_code=404, detail="Run not found")

      log_path = _log_path_for_run(run_id)
      if not log_path.exists():
          return RunLogsResponse(logs=None, source=None)

      return RunLogsResponse(logs=log_path.read_text(), source="local")
  ```

- [ ] **Step 8: Add `stream_run_logs` endpoint**

  Immediately after `get_run_logs`:
  ```python
  @router.get("/{run_id}/logs/stream")
  async def stream_run_logs(
      run_id: str,
      run_store: RunStorePort = Depends(get_run_store),
      user: UserContext = Depends(require_approved),
  ) -> StreamingResponse:
      run = run_store.get(run_id)
      if run is None:
          raise HTTPException(status_code=404, detail="Run not found")
      if run.owner_id is not None and run.owner_id != user.user_id:
          raise HTTPException(status_code=404, detail="Run not found")

      log_path = _log_path_for_run(run_id)

      async def _event_generator():
          sent_bytes = 0
          while True:
              current_run = run_store.get(run_id)
              is_terminal = current_run is None or current_run.status in _TERMINAL_STATUSES

              if log_path.exists():
                  with open(log_path, "rb") as f:
                      f.seek(sent_bytes)
                      chunk = f.read()
                  if chunk:
                      sent_bytes += len(chunk)
                      new_text = chunk.decode("utf-8", errors="replace")
                      for line in new_text.splitlines():
                          yield f"data: {line}\n\n"

              if is_terminal:
                  yield "event: done\ndata: stream closed\n\n"
                  return

              try:
                  await asyncio.sleep(1.0)
              except asyncio.CancelledError:
                  return

      return StreamingResponse(
          _event_generator(),
          media_type="text/event-stream",
          headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
      )
  ```

- [ ] **Step 9: Run integration tests**

  ```bash
  uv run pytest tests/integration/test_log_stream_api.py -v
  ```
  Expected: all PASS.

- [ ] **Step 10: Commit**

  ```bash
  git add src/interactors/api/routes/runs.py tests/integration/test_log_stream_api.py
  git commit -m "feat: add GET /runs/{id}/logs and /logs/stream endpoints"
  ```

---

### Task 5 — Create `useLogStream` hook

**Files:**
- Create: `ui/src/hooks/useLogStream.ts`

Uses `fetch()` instead of `EventSource` so it can send the Auth0 Bearer token via `Authorization` header. When auth is disabled (`VITE_AUTH_DISABLED=true`), `getAuthToken()` returns `null` and the request is sent without a token — which also works.

- [ ] **Step 1: Create the file**

  Create `ui/src/hooks/useLogStream.ts`:
  ```typescript
  import { useEffect, useState } from 'react'
  import { getAuthToken } from '@/api/client'

  interface UseLogStreamResult {
    lines: string[]
  }

  export function useLogStream(runId: string, active: boolean): UseLogStreamResult {
    const [lines, setLines] = useState<string[]>([])

    useEffect(() => {
      if (!active) return

      setLines([])
      let cancelled = false

      async function stream() {
        const token = await getAuthToken()
        const headers: Record<string, string> = {}
        if (token) headers['Authorization'] = `Bearer ${token}`

        let res: Response
        try {
          res = await fetch(`/api/runs/${runId}/logs/stream`, { headers })
        } catch {
          return
        }
        if (!res.ok || !res.body) return

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (!cancelled) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const parts = buffer.split('\n\n')
          buffer = parts.pop() ?? ''
          for (const part of parts) {
            if (part.includes('event: done')) {
              reader.cancel()
              return
            }
            const match = part.match(/^data: (.*)$/m)
            if (match) {
              setLines(prev => [...prev, match[1]])
            }
          }
        }
        reader.cancel()
      }

      stream().catch(() => {})

      return () => {
        cancelled = true
      }
    }, [runId, active])

    return { lines }
  }
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add ui/src/hooks/useLogStream.ts
  git commit -m "feat: add useLogStream hook (Fetch-based SSE with Bearer auth)"
  ```

---

### Task 6 — Wire `RunLogViewer` into `RunDetailPage`

**Files:**
- Modify: `ui/src/pages/RunDetailPage.tsx`

- [ ] **Step 1: Add imports**

  In `ui/src/pages/RunDetailPage.tsx`, add after the existing imports:
  ```typescript
  import { RunLogViewer } from '@/components/RunLogViewer'
  import { useLogStream } from '@/hooks/useLogStream'
  ```
  Also add `getRunLogs` to the existing runs import on line 5:
  ```typescript
  import { cancelRun, deleteRun, getRunEvaluation, getRun, getRunLogs, isRunActive, isRunCancellable } from '@/api/runs'
  ```

- [ ] **Step 2: Add log data before the early returns**

  After the `cancelMutation` declaration and **before** the `if (isLoading)` early return, insert these lines. React hooks must be called unconditionally — placing them here satisfies that rule. When `run` is undefined (loading), `runIsActive` is `false` so the hook and query stay inactive.

  ```typescript
  const runIsActive = run != null && isRunActive(run)
  const { lines } = useLogStream(runId!, runIsActive)

  const { data: logsData } = useQuery({
    queryKey: ['runs', runId, 'logs'],
    queryFn: () => getRunLogs(runId!),
    enabled: run != null && !runIsActive,
  })

  const logContent = runIsActive ? lines.join('\n') : (logsData?.logs ?? '')
  ```

- [ ] **Step 3: Add the log viewer section to JSX**

  After the closing `</section>` of the Metrics grid (after the metrics `</dl>` and `</section>` around line 214), add:
  ```tsx
  <hr className="ed-rule" />
  <section className="mb-10">
    <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a] mb-4">
      Workflow logs
    </h2>
    <RunLogViewer logs={logContent} />
  </section>
  ```

- [ ] **Step 4: Verify TypeScript compiles**

  ```bash
  cd ui && npx tsc --noEmit 2>&1
  ```
  Expected: no errors.

- [ ] **Step 5: Commit**

  ```bash
  git add ui/src/pages/RunDetailPage.tsx
  git commit -m "feat: display workflow logs on run detail page"
  ```

---

## Verification (end-to-end)

1. Start backend: `uv run uvicorn interactors.api.app:app --reload --port 8000`
2. Start UI: `cd ui && npm run dev`
3. Navigate to `/runs`, click a run → the Run Detail page now has a **"Workflow logs"** section
4. For an **active** run: log lines stream in live (1-second poll from the SSE endpoint)
5. For a **completed/failed** run: the full `data/workflow/{run_id}/logs.txt` content appears on page load
6. For the **deployed site**: confirm `VITE_API_URL=https://api.llm.jwnwilson.co.uk` is set in the deployment environment, or that nginx proxies `/api` to the backend — then rebuild and redeploy

> **Auth note on SSE:** `useLogStream` uses `fetch()` which sends the Bearer token via `Authorization` header. If you see 401 errors on the stream endpoint, confirm `setTokenGetter` is called before any active run is viewed (it's wired in `App.tsx`).
