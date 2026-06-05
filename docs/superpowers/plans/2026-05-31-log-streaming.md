# Log Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream training run logs to the UI in real-time using Server-Sent Events (SSE), replacing the current poll-every-5s fetch with a live tail of the log file while the run is active.

**Architecture:** Add `GET /api/runs/{run_id}/logs/stream` SSE endpoint using `sse-starlette` that tails `data/workflow/{run_id}/logs.txt` and pushes new lines as events. The stream closes when the run reaches a terminal status. On the frontend, a `useLogStream` hook opens an `EventSource` while the run is active; `RunDetailsPanel` uses it for active runs and falls back to the existing REST fetch for completed/failed runs.

**Tech Stack:** Python/FastAPI + `sse-starlette` (backend SSE), React/TypeScript + browser `EventSource` API, Vitest (frontend tests), pytest + httpx (backend integration tests).

---

## File Map

**Create:**
- `ui/src/hooks/useLogStream.ts`
- `ui/src/test/hooks/useLogStream.test.ts`
- `tests/integration/test_log_stream_api.py`

**Modify:**
- `pyproject.toml` — add `sse-starlette`
- `src/interactors/api/routes/runs.py` — add `GET /{run_id}/logs/stream` endpoint
- `ui/src/components/RunDetailsPanel.tsx` — use `useLogStream` for active runs
- `ui/src/test/msw/handlers.ts` — add SSE handler stub

---

## Task 1: Add `sse-starlette` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

```bash
uv add sse-starlette
```

- [ ] **Step 2: Verify it installed**

```bash
uv run python -c "import sse_starlette; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add sse-starlette dependency"
```

---

## Task 2: Add the SSE log streaming endpoint

**Files:**
- Modify: `src/interactors/api/routes/runs.py`
- Create: `tests/integration/test_log_stream_api.py`

- [ ] **Step 1: Check existing integration conftest to understand fixtures**

```bash
cat tests/integration/conftest.py
```

This file provides the `client` fixture (authenticated `TestClient`) and likely a `run_store` fixture. Use the same fixtures in the new test file.

- [ ] **Step 2: Write the failing test**

```python
# tests/integration/test_log_stream_api.py
"""Integration tests for the SSE log streaming endpoint."""
from __future__ import annotations

import pytest
from pathlib import Path


def test_log_stream_404_for_unknown_run(client):
    """Unauthenticated or nonexistent run returns 404."""
    resp = client.get("/api/runs/no-such-run/logs/stream")
    assert resp.status_code == 404


def test_log_stream_returns_text_event_stream(client, run_store, monkeypatch, tmp_path):
    """A completed run with a log file streams SSE content-type."""
    from domain.models import RunConfig, RunStatus

    run = run_store.create(RunConfig(model_id="m1", workflow_id="wf-stream-test"))
    run_store.update_status(run.id, RunStatus.COMPLETED)

    log_file = tmp_path / "logs.txt"
    log_file.write_text("epoch 1 done\nepoch 2 done\n", encoding="utf-8")

    monkeypatch.setattr(
        "interactors.api.routes.runs._log_path_for_run",
        lambda rid: log_file,
    )

    with client.stream("GET", f"/api/runs/{run.id}/logs/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        collected = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                collected.append(line)
            if "done" in line and line.startswith("event:"):
                break

    assert any("epoch 1 done" in l for l in collected)
    assert any("epoch 2 done" in l for l in collected)
```

- [ ] **Step 3: Run to verify failure**

```bash
uv run pytest tests/integration/test_log_stream_api.py::test_log_stream_404_for_unknown_run -v
```
Expected: 404 (FastAPI returns 404 for unmatched routes) — test will fail because the route doesn't exist yet. Once we add it, 404 comes from our logic.

- [ ] **Step 4: Add the SSE endpoint to `src/interactors/api/routes/runs.py`**

Add these imports at the top of `runs.py` (alongside existing imports):

```python
import asyncio
from fastapi.responses import StreamingResponse
```

Add a module-level helper function after the `log = logging.getLogger(__name__)` line:

```python
def _log_path_for_run(run_id: str) -> Path:
    return Path(f"data/workflow/{run_id}/logs.txt")
```

Add the new endpoint after the existing `get_run_logs` endpoint:

```python
_TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})


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
                content = log_path.read_text(encoding="utf-8", errors="replace")
                if len(content) > sent_bytes:
                    new_text = content[sent_bytes:]
                    sent_bytes = len(content)
                    for line in new_text.splitlines():
                        yield f"data: {line}\n\n"

            if is_terminal:
                yield "event: done\ndata: stream closed\n\n"
                return

            await asyncio.sleep(1.0)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 5: Run integration tests**

```bash
uv run pytest tests/integration/test_log_stream_api.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Verify no regressions**

```bash
uv run pytest tests/unit/ tests/integration/ -v --ignore=tests/integration/test_real_inference.py -x
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/interactors/api/routes/runs.py tests/integration/test_log_stream_api.py
git commit -m "feat: add SSE log streaming endpoint GET /api/runs/{id}/logs/stream"
```

---

## Task 3: Create `useLogStream` hook with tests

**Files:**
- Create: `ui/src/hooks/useLogStream.ts`
- Create: `ui/src/test/hooks/useLogStream.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// ui/src/test/hooks/useLogStream.test.ts
import { renderHook, act } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { useLogStream } from '@/hooks/useLogStream'

class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  onmessage: ((e: { data: string }) => void) | null = null
  addEventListener = vi.fn((event: string, cb: () => void) => {
    if (event === 'done') this._doneHandler = cb
  })
  close = vi.fn()
  _doneHandler: (() => void) | null = null

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  emit(data: string) {
    this.onmessage?.({ data })
  }
}

beforeEach(() => {
  MockEventSource.instances = []
  vi.stubGlobal('EventSource', MockEventSource)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useLogStream', () => {
  it('returns empty lines when active=false', () => {
    const { result } = renderHook(() => useLogStream('run-1', false))
    expect(result.current.lines).toEqual([])
  })

  it('does not open EventSource when active=false', () => {
    renderHook(() => useLogStream('run-1', false))
    expect(MockEventSource.instances).toHaveLength(0)
  })

  it('opens EventSource when active=true', () => {
    renderHook(() => useLogStream('run-1', true))
    expect(MockEventSource.instances).toHaveLength(1)
    expect(MockEventSource.instances[0].url).toContain('/api/runs/run-1/logs/stream')
  })

  it('accumulates lines from onmessage events', () => {
    const { result } = renderHook(() => useLogStream('run-1', true))
    act(() => { MockEventSource.instances[0].emit('epoch 1') })
    act(() => { MockEventSource.instances[0].emit('epoch 2') })
    expect(result.current.lines).toEqual(['epoch 1', 'epoch 2'])
  })

  it('closes EventSource on unmount', () => {
    const { unmount } = renderHook(() => useLogStream('run-1', true))
    unmount()
    expect(MockEventSource.instances[0].close).toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd ui && npx vitest run src/test/hooks/useLogStream.test.ts
```
Expected: fail — module not found.

- [ ] **Step 3: Create `ui/src/hooks/useLogStream.ts`**

```typescript
import { useEffect, useState } from 'react'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

interface UseLogStreamResult {
  lines: string[]
}

export function useLogStream(runId: string, active: boolean): UseLogStreamResult {
  const [lines, setLines] = useState<string[]>([])

  useEffect(() => {
    if (!active) return

    const es = new EventSource(`${API_BASE}/api/runs/${runId}/logs/stream`)

    es.onmessage = (e: MessageEvent<string>) => {
      setLines(prev => [...prev, e.data])
    }

    es.addEventListener('done', () => {
      es.close()
    })

    return () => {
      es.close()
    }
  }, [runId, active])

  return { lines }
}
```

- [ ] **Step 4: Run tests**

```bash
npx vitest run src/test/hooks/useLogStream.test.ts
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/src/hooks/useLogStream.ts ui/src/test/hooks/useLogStream.test.ts
git commit -m "feat: add useLogStream hook for SSE log tailing"
```

---

## Task 4: Wire `useLogStream` into `RunDetailsPanel`

**Files:**
- Modify: `ui/src/components/RunDetailsPanel.tsx`
- Modify: `ui/src/test/msw/handlers.ts`

- [ ] **Step 1: Read `RunDetailsPanel.tsx` to understand current structure**

Note that `isRunActive(run)` currently returns `run.status === 'running'`. For log streaming we want to stream for ALL non-terminal statuses. Check `src/api/runs.ts` — the `ACTIVE_STATUSES` set there covers `pending`, `generating`, `training`, `evaluating`, `exporting`, `running`. Export a helper or duplicate the check locally.

In `ui/src/api/runs.ts`, export the active statuses check:

```typescript
export const ACTIVE_STATUSES = new Set<RunStatus>([
  'pending', 'generating', 'training', 'evaluating', 'exporting', 'running',
])

export function isRunActive(run: RunRecord): boolean {
  return ACTIVE_STATUSES.has(run.status)
}
```

- [ ] **Step 2: Update `RunDetailsPanel.tsx`**

Add import:

```typescript
import { useLogStream } from '@/hooks/useLogStream'
import { isRunActive } from '@/api/runs'
```

Inside the component body, after the `pollInterval` line:

```typescript
const isActive = isRunActive(run)
const { lines: streamedLines } = useLogStream(runId, isActive && expanded)
```

Update the `getRunLogs` query to only run when the run is not active:

```typescript
const { data: logsData } = useQuery({
  queryKey: ['runs', runId, 'logs'],
  queryFn: () => getRunLogs(runId),
  enabled: expanded && !isActive,
  refetchInterval: false,
})
```

Inside the expanded section, replace the log rendering:

```tsx
{/* Static logs for completed/failed runs */}
{!isActive && logsData != null && <LogsSection logsData={logsData} />}

{/* Live stream for active runs */}
{isActive && (
  <div>
    <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.18em] text-[#888888] mb-2">
      Training logs — live
    </div>
    <pre className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#1a1a1a] bg-[#f6f5f0] border border-[#d0d0c8] rounded-[2px] p-3 overflow-x-auto whitespace-pre-wrap break-all max-h-64">
      {streamedLines.length > 0 ? streamedLines.join('\n') : (
        <span className="text-[#888888] italic font-['DM_Serif_Display']">Waiting for output…</span>
      )}
    </pre>
  </div>
)}
```

- [ ] **Step 3: Add SSE stub to MSW handlers**

In `ui/src/test/msw/handlers.ts`, add:

```typescript
http.get(`${BASE}/api/runs/:id/logs/stream`, () => {
  return new HttpResponse(
    'data: test-log-line\n\nevent: done\ndata: stream closed\n\n',
    { headers: { 'Content-Type': 'text/event-stream' } },
  )
}),
```

- [ ] **Step 4: Run frontend test suite**

```bash
cd ui && npx vitest run
```
Expected: all tests PASS. The `RunDetailsPanel` test renders a completed run (`RUN_FIXTURE` status is `completed`) so `isActive` is `false` there — streaming branch is not exercised, no test changes needed.

- [ ] **Step 5: Commit**

```bash
git add ui/src/components/RunDetailsPanel.tsx ui/src/api/runs.ts ui/src/test/msw/handlers.ts
git commit -m "feat: stream live logs in RunDetailsPanel for active runs"
```

---

## Self-Review

- SSE generator tracks `sent_bytes` so it never re-emits already-sent content — correct.
- Generator exits cleanly on terminal statuses — no goroutine / async task leak.
- `useLogStream` guards with `if (!active) return` — no EventSource opened when not needed.
- Cleanup closes the EventSource on unmount — no browser connection leak.
- Static REST fallback for completed/failed runs is unchanged — backward compatible.
- `isRunActive` updated to use the full `ACTIVE_STATUSES` set — logs stream for all non-terminal phases.
- No `any` types introduced in TypeScript.
- MSW stub added so tests that render RunDetailsPanel (even with an active run) won't error on EventSource.
