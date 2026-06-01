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
