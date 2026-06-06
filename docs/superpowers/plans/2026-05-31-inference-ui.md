# Inference UI — Group by Model & Test Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Inference page so instances are grouped by model name, and add an inline test panel on each available instance so the user can fire a real inference request and see the response without leaving the page.

**Architecture:** No new backend endpoints are needed. The existing `POST /api/inferences/{instance_id}/infer` handles test calls. The frontend fetches both `/api/models` and `/api/inferences`, builds a `Map<model_id, { model, instances[] }>` client-side, and renders one collapsible section per model. Each available instance has a "Test" toggle that expands an `InstanceInferencePanel` with a JSON request editor and response display.

**Tech Stack:** React/TypeScript + TanStack Query, Vitest + MSW (frontend tests). No backend changes required.

---

## File Map

**Create:**
- `ui/src/components/InstanceInferencePanel.tsx`
- `ui/src/test/components/InstanceInferencePanel.test.tsx`
- `ui/src/test/api/inferences.test.ts` (if it does not exist)

**Modify:**
- `ui/src/api/inferences.ts` — add `inferInstance()` function
- `ui/src/pages/InferencePage.tsx` — group by model, show `InstanceInferencePanel`
- `ui/src/test/msw/handlers.ts` — add `POST /api/inferences/:id/infer` handler
- `ui/src/test/pages/InferencePage.test.tsx` — cover grouped layout

---

## Task 1: Add `inferInstance` to the inferences API client

**Files:**
- Modify: `ui/src/api/inferences.ts`

- [ ] **Step 1: Read `ui/src/api/inferences.ts`**

Confirm existing exports and imports before editing.

- [ ] **Step 2: Check if `ui/src/test/api/inferences.test.ts` exists**

```bash
ls ui/src/test/api/
```

If it does not exist, create it. If it does, append the new test.

- [ ] **Step 3: Write the failing test**

```typescript
// ui/src/test/api/inferences.test.ts
import { describe, it, expect } from 'vitest'
import { inferInstance } from '@/api/inferences'
import { server } from '@/test/msw/server'
import { http, HttpResponse } from 'msw'

const BASE = 'http://localhost:8000'

describe('inferInstance', () => {
  it('posts to the correct endpoint and returns the response', async () => {
    const mockResponse = { action: 'EAT', stat: null, target_object_id: 'bowl-1', confidence: 0.9 }
    server.use(
      http.post(`${BASE}/api/inferences/inst-1/infer`, () => HttpResponse.json(mockResponse)),
    )
    const req = {
      scene: { objects: [{ type: 'bowl' as const, id: 'bowl-1', distance: 2.0 }], tick: 1 },
      pet_stats: { hunger: 0.8, tiredness: 0.1, boredom: 0.2, social: 0.0, toilet: 0.0 },
    }
    const result = await inferInstance('inst-1', req)
    expect(result.action).toBe('EAT')
    expect(result.target_object_id).toBe('bowl-1')
  })
})
```

- [ ] **Step 4: Run to verify failure**

```bash
cd ui && npx vitest run src/test/api/inferences.test.ts
```
Expected: fail — `inferInstance` is not exported.

- [ ] **Step 5: Add `inferInstance` to `ui/src/api/inferences.ts`**

Append after the existing `deleteInference` function:

```typescript
import type { InferenceInstance, InferenceRequest, InferenceResponse, PaginatedResponse } from '@/types'

export async function inferInstance(id: string, request: InferenceRequest): Promise<InferenceResponse> {
  const { data } = await apiClient.post<InferenceResponse>(`/api/inferences/${id}/infer`, request)
  return data
}
```

- [ ] **Step 6: Run test to verify it passes**

```bash
npx vitest run src/test/api/inferences.test.ts
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ui/src/api/inferences.ts ui/src/test/api/inferences.test.ts
git commit -m "feat: add inferInstance API client function"
```

---

## Task 2: Create `InstanceInferencePanel` component

**Files:**
- Create: `ui/src/components/InstanceInferencePanel.tsx`
- Create: `ui/src/test/components/InstanceInferencePanel.test.tsx`

- [ ] **Step 1: Write the failing tests**

```typescript
// ui/src/test/components/InstanceInferencePanel.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/msw/server'
import { InstanceInferencePanel } from '@/components/InstanceInferencePanel'

const BASE = 'http://localhost:8000'

function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {children}
    </QueryClientProvider>
  )
}

describe('InstanceInferencePanel', () => {
  it('renders the run inference button', () => {
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    expect(screen.getByRole('button', { name: /run inference/i })).toBeInTheDocument()
  })

  it('renders a textarea with default payload containing pet_stats', () => {
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    const textarea = screen.getByRole('textbox')
    expect(textarea).toHaveValue(expect.stringContaining('pet_stats'))
  })

  it('shows error on invalid JSON before submitting', async () => {
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    await userEvent.clear(screen.getByRole('textbox'))
    await userEvent.type(screen.getByRole('textbox'), 'not json')
    await userEvent.click(screen.getByRole('button', { name: /run inference/i }))
    expect(screen.getByText(/invalid json/i)).toBeInTheDocument()
  })

  it('displays inference result action on success', async () => {
    server.use(
      http.post(`${BASE}/api/inferences/inst-1/infer`, () =>
        HttpResponse.json({ action: 'SLEEP', stat: null, target_object_id: 'bed-1', confidence: 0.85 }),
      ),
    )
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    await userEvent.click(screen.getByRole('button', { name: /run inference/i }))
    await waitFor(() => expect(screen.getByText('SLEEP')).toBeInTheDocument())
  })

  it('displays error message on API failure', async () => {
    server.use(
      http.post(`${BASE}/api/inferences/inst-1/infer`, () =>
        HttpResponse.json({ detail: 'not available' }, { status: 409 }),
      ),
    )
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    await userEvent.click(screen.getByRole('button', { name: /run inference/i }))
    await waitFor(() => expect(screen.getByText(/inference failed/i)).toBeInTheDocument())
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd ui && npx vitest run src/test/components/InstanceInferencePanel.test.tsx
```
Expected: fail — component does not exist.

- [ ] **Step 3: Create `ui/src/components/InstanceInferencePanel.tsx`**

```typescript
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Zap } from 'lucide-react'
import { inferInstance } from '@/api/inferences'
import { Button } from '@/components/ui/button'
import type { InferenceRequest, InferenceResponse } from '@/types'

const DEFAULT_REQUEST: InferenceRequest = {
  scene: {
    objects: [
      { type: 'bowl', id: 'bowl-1', distance: 2.5 },
      { type: 'toy', id: 'toy-1', distance: 4.0 },
    ],
    tick: 1,
  },
  pet_stats: {
    hunger: 0.8,
    tiredness: 0.2,
    boredom: 0.3,
    social: 0.1,
    toilet: 0.0,
  },
}

interface InstanceInferencePanelProps {
  instanceId: string
}

export function InstanceInferencePanel({ instanceId }: InstanceInferencePanelProps) {
  const [json, setJson] = useState(JSON.stringify(DEFAULT_REQUEST, null, 2))
  const [parseError, setParseError] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: (req: InferenceRequest) => inferInstance(instanceId, req),
  })

  function handleRun() {
    setParseError(null)
    let parsed: InferenceRequest
    try {
      parsed = JSON.parse(json) as InferenceRequest
    } catch {
      setParseError('Invalid JSON')
      return
    }
    mutation.mutate(parsed)
  }

  return (
    <div className="mt-3 pt-3 border-t border-[#e5e3d8]">
      <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.14em] text-[#888888] mb-2">
        Test inference
      </div>

      <textarea
        className="w-full bg-white px-3 py-2 font-['IBM_Plex_Mono'] text-[0.75rem] text-[#1a1a1a] min-h-36 resize-y border-[1.5px] border-[#d0d0c8] rounded-[3px] focus:outline-none focus:border-[#1a1a1a]"
        value={json}
        onChange={e => setJson(e.target.value)}
        spellCheck={false}
        aria-label="Inference request payload"
      />

      {parseError && (
        <p className="font-['IBM_Plex_Mono'] text-[0.72rem] uppercase tracking-[0.12em] text-[#7f1d1d] mt-1">
          {parseError}
        </p>
      )}

      <div className="mt-2">
        <Button size="sm" onClick={handleRun} disabled={mutation.isPending}>
          <Zap className="h-3 w-3" />
          {mutation.isPending ? 'Running…' : 'Run inference'}
        </Button>
      </div>

      {mutation.isError && (
        <div className="mt-2 border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-3 py-2">
          <p className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#7f1d1d]">
            Inference failed: {String((mutation.error as Error)?.message ?? 'unknown error')}
          </p>
        </div>
      )}

      {mutation.isSuccess && <InferenceResult result={mutation.data} />}
    </div>
  )
}

function InferenceResult({ result }: { result: InferenceResponse }) {
  return (
    <div className="mt-2 bg-[#f6f5ef] border-l-[3px] border-[#1a1a1a] px-4 py-3">
      <div className="font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em] text-[#888888] mb-2">
        Response
      </div>
      <dl className="grid grid-cols-3 gap-x-4 gap-y-1">
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.12em] text-[#888888]">Action</dt>
          <dd className="font-['DM_Serif_Display'] text-[1rem] text-[#1a1a1a]">{result.action}</dd>
        </div>
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.12em] text-[#888888]">Target</dt>
          <dd className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">{result.target_object_id ?? '—'}</dd>
        </div>
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.12em] text-[#888888]">Confidence</dt>
          <dd className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">
            {result.confidence != null ? `${(result.confidence * 100).toFixed(0)}%` : '—'}
          </dd>
        </div>
      </dl>
    </div>
  )
}
```

- [ ] **Step 4: Run tests**

```bash
npx vitest run src/test/components/InstanceInferencePanel.test.tsx
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/src/components/InstanceInferencePanel.tsx ui/src/test/components/InstanceInferencePanel.test.tsx
git commit -m "feat: add InstanceInferencePanel component with tests"
```

---

## Task 3: Add MSW handler for `POST /api/inferences/:id/infer`

**Files:**
- Modify: `ui/src/test/msw/handlers.ts`

- [ ] **Step 1: Add the default handler**

In `ui/src/test/msw/handlers.ts`, add before the closing `]` of the `handlers` array:

```typescript
http.post(`${BASE}/api/inferences/:id/infer`, async () => {
  return HttpResponse.json({
    action: 'EAT',
    stat: null,
    target_object_id: 'bowl-1',
    confidence: 0.92,
  })
}),
```

- [ ] **Step 2: Run frontend tests to verify no breakage**

```bash
cd ui && npx vitest run
```
Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add ui/src/test/msw/handlers.ts
git commit -m "test: add MSW handler for instance infer endpoint"
```

---

## Task 4: Redesign `InferencePage` — group by model, add test panel

**Files:**
- Modify: `ui/src/pages/InferencePage.tsx`
- Modify or create: `ui/src/test/pages/InferencePage.test.tsx`

- [ ] **Step 1: Read `ui/src/pages/InferencePage.tsx` and the existing test file**

```bash
cat ui/src/pages/InferencePage.tsx
ls ui/src/test/pages/
```

- [ ] **Step 2: Write tests for the grouped layout**

If `ui/src/test/pages/InferencePage.test.tsx` does not exist, create it. If it does, read it and append:

```typescript
// ui/src/test/pages/InferencePage.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/msw/server'
import { InferencePage } from '@/pages/InferencePage'
import { resetHandlerState } from '@/test/msw/handlers'

const BASE = 'http://localhost:8000'

function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <MemoryRouter>
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        {children}
      </QueryClientProvider>
    </MemoryRouter>
  )
}

const MODEL = {
  id: 'm1', name: 'My Model', description: '', base_model: 'base',
  train_data: 't.jsonl', eval_data: 'e.jsonl', epochs: 5, patience: 3,
  warmup_ratio: 0.05, remote_backend: 'local', skip_generate: false,
  created_at: '2024-01-01T00:00:00Z', updated_at: '2024-01-01T00:00:00Z',
}

const AVAILABLE_INSTANCE = {
  id: 'inst-1', model_id: 'm1', pod_name: 'pod-abc', pod_namespace: 'default',
  idle_timeout_minutes: 120, status: 'available',
  last_used_at: null, created_at: '2024-01-01T00:00:00Z', updated_at: '2024-01-01T00:00:00Z',
}

function paged<T>(items: T[]) {
  return { items, total: items.length, page: 1, limit: 50, pages: 1 }
}

beforeEach(() => resetHandlerState())

describe('InferencePage', () => {
  it('shows model name as section heading', async () => {
    server.use(
      http.get(`${BASE}/api/models`, () => HttpResponse.json(paged([MODEL]))),
      http.get(`${BASE}/api/inferences`, () => HttpResponse.json(paged([AVAILABLE_INSTANCE]))),
    )
    render(<InferencePage />, { wrapper })
    await waitFor(() => expect(screen.getByText('My Model')).toBeInTheDocument())
  })

  it('shows pod name under the model heading', async () => {
    server.use(
      http.get(`${BASE}/api/models`, () => HttpResponse.json(paged([MODEL]))),
      http.get(`${BASE}/api/inferences`, () => HttpResponse.json(paged([AVAILABLE_INSTANCE]))),
    )
    render(<InferencePage />, { wrapper })
    await waitFor(() => expect(screen.getByText('pod-abc')).toBeInTheDocument())
  })

  it('shows Test button for available instance', async () => {
    server.use(
      http.get(`${BASE}/api/models`, () => HttpResponse.json(paged([MODEL]))),
      http.get(`${BASE}/api/inferences`, () => HttpResponse.json(paged([AVAILABLE_INSTANCE]))),
    )
    render(<InferencePage />, { wrapper })
    await waitFor(() => expect(screen.getByRole('button', { name: /test/i })).toBeInTheDocument())
  })

  it('expands inference test panel on Test click', async () => {
    server.use(
      http.get(`${BASE}/api/models`, () => HttpResponse.json(paged([MODEL]))),
      http.get(`${BASE}/api/inferences`, () => HttpResponse.json(paged([AVAILABLE_INSTANCE]))),
    )
    render(<InferencePage />, { wrapper })
    await waitFor(() => screen.getByRole('button', { name: /test/i }))
    await userEvent.click(screen.getByRole('button', { name: /test/i }))
    expect(screen.getByRole('button', { name: /run inference/i })).toBeInTheDocument()
  })

  it('shows empty state when no instances', async () => {
    server.use(
      http.get(`${BASE}/api/models`, () => HttpResponse.json(paged([]))),
      http.get(`${BASE}/api/inferences`, () => HttpResponse.json(paged([]))),
    )
    render(<InferencePage />, { wrapper })
    await waitFor(() =>
      expect(screen.getByText(/no inference instances/i)).toBeInTheDocument()
    )
  })
})
```

- [ ] **Step 3: Run to verify failure**

```bash
cd ui && npx vitest run src/test/pages/InferencePage.test.tsx
```
Expected: fail — current `InferencePage` renders a flat table, no model grouping.

- [ ] **Step 4: Rewrite `ui/src/pages/InferencePage.tsx`**

Replace the entire file contents:

```typescript
import { useState, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, ChevronUp, Play, Square, Trash2 } from 'lucide-react'
import { deleteInference, listInferences, startInference, stopInference } from '@/api/inferences'
import { listModels } from '@/api/models'
import { InferenceStatusBadge } from '@/components/InferenceStatusBadge'
import { InstanceInferencePanel } from '@/components/InstanceInferencePanel'
import { Button } from '@/components/ui/button'
import type { InferenceInstance, InferenceStatus, TrainingModel } from '@/types'

const AUTO_REFRESH_MS = 10_000
const CAN_START: InferenceStatus[] = ['pending', 'shutdown', 'failed']
const CAN_STOP: InferenceStatus[] = ['available', 'initializing', 'idle']
const CAN_DELETE: InferenceStatus[] = ['pending', 'shutdown', 'failed']

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

interface ModelGroupProps {
  model: TrainingModel | null
  instances: InferenceInstance[]
  onStart: (id: string) => void
  onStop: (id: string) => void
  onDelete: (id: string) => void
  pendingStart: string | null
  pendingStop: string | null
  pendingDelete: string | null
}

function ModelGroup({
  model, instances, onStart, onStop, onDelete,
  pendingStart, pendingStop, pendingDelete,
}: ModelGroupProps) {
  const [expandedTest, setExpandedTest] = useState<string | null>(null)

  return (
    <section className="mb-8">
      <div className="flex items-baseline gap-3 mb-3">
        <h2 className="font-['DM_Serif_Display'] text-[1.3rem] text-[#1a1a1a]">
          {model?.name ?? 'Unknown model'}
        </h2>
        <span className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888]">
          {instances.length} instance{instances.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
        {instances.map((instance, i) => (
          <div key={instance.id} className={i < instances.length - 1 ? 'border-b border-[#e5e3d8]' : ''}>
            <div className="flex items-center justify-between gap-4 px-5 py-4 flex-wrap">
              <div className="flex items-center gap-6 min-w-0">
                <InferenceStatusBadge status={instance.status} />
                <div>
                  <p className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">
                    {instance.pod_name || '—'}
                  </p>
                  <p className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#888888] mt-0.5">
                    Last used: {formatDate(instance.last_used_at)}
                  </p>
                </div>
              </div>

              <div className="flex gap-2 items-center shrink-0">
                {instance.status === 'available' && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setExpandedTest(prev => prev === instance.id ? null : instance.id)}
                    aria-label="Test inference"
                  >
                    {expandedTest === instance.id
                      ? <ChevronUp className="h-3 w-3" />
                      : <ChevronDown className="h-3 w-3" />}
                    Test
                  </Button>
                )}
                {CAN_START.includes(instance.status) && (
                  <Button size="sm" onClick={() => onStart(instance.id)}
                    disabled={pendingStart === instance.id}
                    aria-label={`Start ${instance.id}`}>
                    <Play className="h-3 w-3" />Start
                  </Button>
                )}
                {CAN_STOP.includes(instance.status) && (
                  <Button size="sm" variant="outline" onClick={() => onStop(instance.id)}
                    disabled={pendingStop === instance.id}
                    aria-label={`Stop ${instance.id}`}>
                    <Square className="h-3 w-3" />Stop
                  </Button>
                )}
                {CAN_DELETE.includes(instance.status) && (
                  <Button size="sm" variant="destructive" onClick={() => onDelete(instance.id)}
                    disabled={pendingDelete === instance.id}
                    aria-label={`Delete ${instance.id}`}>
                    <Trash2 className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>

            {expandedTest === instance.id && (
              <div className="px-5 pb-4">
                <InstanceInferencePanel instanceId={instance.id} />
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

export function InferencePage() {
  const queryClient = useQueryClient()

  const { data: instancesData, isLoading, isError } = useQuery({
    queryKey: ['inferences'],
    queryFn: () => listInferences(),
  })
  const instances = instancesData?.items ?? []

  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: () => listModels(),
  })
  const modelsById = new Map<string, TrainingModel>(
    (modelsData?.items ?? []).map(m => [m.id, m]),
  )

  useEffect(() => {
    const interval = setInterval(
      () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
      AUTO_REFRESH_MS,
    )
    return () => clearInterval(interval)
  }, [queryClient])

  const startMutation = useMutation({
    mutationFn: (id: string) => startInference(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })
  const stopMutation = useMutation({
    mutationFn: (id: string) => stopInference(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteInference(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })

  const grouped = new Map<string, InferenceInstance[]>()
  for (const inst of instances) {
    const list = grouped.get(inst.model_id) ?? []
    grouped.set(inst.model_id, [...list, inst])
  }

  if (isLoading) {
    return (
      <div className="ed-page">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
          Loading instances
        </span>
      </div>
    )
  }
  if (isError) {
    return (
      <div className="ed-page">
        <div className="border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-4 py-3 inline-block">
          <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
            Failed to load inference instances.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="ed-page">
      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888] mb-3">
          Vol. 4 · Runtime
        </div>
        <h1 className="font-['DM_Serif_Display'] text-[2.4rem] leading-[1.05] text-[#1a1a1a] mb-3">
          Inference instances
        </h1>
        <p className="font-['Outfit'] text-[1rem] text-[#3a3a36] max-w-2xl leading-relaxed">
          Inference pods grouped by model. Start an instance to make a model available for serving;
          use the Test panel to run a sample request against any available pod.
        </p>
        <hr className="ed-rule mt-7 mb-0" />
      </header>

      {grouped.size === 0 ? (
        <div className="border border-dashed border-[#d0d0c8] bg-white/40 rounded-[4px] py-16 text-center">
          <p className="font-['DM_Serif_Display'] italic text-[1.4rem] text-[#3a3a36] mb-1">
            No inference instances.
          </p>
          <p className="font-['Outfit'] text-[0.9rem] text-[#888888]">
            Train a model to provision a serving instance.
          </p>
        </div>
      ) : (
        Array.from(grouped.entries()).map(([modelId, modelInstances]) => (
          <ModelGroup
            key={modelId}
            model={modelsById.get(modelId) ?? null}
            instances={modelInstances}
            onStart={id => startMutation.mutate(id)}
            onStop={id => stopMutation.mutate(id)}
            onDelete={id => deleteMutation.mutate(id)}
            pendingStart={startMutation.isPending ? (startMutation.variables ?? null) : null}
            pendingStop={stopMutation.isPending ? (stopMutation.variables ?? null) : null}
            pendingDelete={deleteMutation.isPending ? (deleteMutation.variables ?? null) : null}
          />
        ))
      )}
    </div>
  )
}
```

- [ ] **Step 5: Run the grouped layout tests**

```bash
cd ui && npx vitest run src/test/pages/InferencePage.test.tsx
```
Expected: all 5 tests PASS.

- [ ] **Step 6: Run the full frontend test suite**

```bash
npx vitest run
```
Expected: all tests PASS. If old `InferencePage` snapshot or structural tests exist and now fail, update them to match the new grouped layout.

- [ ] **Step 7: Commit**

```bash
git add ui/src/pages/InferencePage.tsx ui/src/test/pages/InferencePage.test.tsx
git commit -m "feat: redesign InferencePage — group by model, add per-instance inference test panel"
```

---

## Self-Review

- `InferencePage` fetches both models and inferences — grouping is purely client-side, no new endpoint.
- `InstanceInferencePanel` is only shown (and the Test button only visible) for `available` instances — prevents wasted calls to pods that can't serve.
- Test panel is collapsible: one open at a time per model group via `expandedTest` state.
- `inferInstance` posts to the existing `POST /api/inferences/{id}/infer` — no backend changes needed.
- All mutations invalidate `['inferences']` so the list refreshes after start/stop/delete.
- Immutable patterns: new arrays created with spread (`[...list, inst]`), new Map from array spread.
- No `any` types introduced.
- MSW default handler returns a plausible success response so other tests don't break.
- If the pagination plan is merged first, `listInferences()` and `listModels()` already return `PaginatedResponse` — use `.items` to extract the array (as shown above with `instancesData?.items ?? []`).
