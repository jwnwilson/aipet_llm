import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/msw/server'
import { ModelDetailPage } from '@/pages/ModelDetailPage'
import { MODEL_FIXTURE } from '@/test/msw/fixtures'
import { resetHandlerState } from '@/test/msw/handlers'

const BASE = 'http://localhost:8000'
const MODEL_WITH_GGUF = { ...MODEL_FIXTURE, gguf_path: 'model/abc.gguf' }

function renderPage(modelId: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/models/${modelId}`]}>
        <Routes>
          <Route path="/models/:id" element={<ModelDetailPage />} />
          <Route path="/inferences" element={<div>inferences-page</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => resetHandlerState())

describe('ModelDetailPage', () => {
  it('shows Deploy button when model has a gguf_path', async () => {
    server.use(
      http.get(`${BASE}/api/models/:id`, () => HttpResponse.json(MODEL_WITH_GGUF)),
    )
    renderPage(MODEL_WITH_GGUF.id)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /deploy/i })).toBeInTheDocument(),
    )
  })

  it('does not show Deploy button when model has no gguf_path', async () => {
    server.use(
      http.get(`${BASE}/api/models/:id`, () => HttpResponse.json(MODEL_FIXTURE)),
    )
    renderPage(MODEL_FIXTURE.id)
    await waitFor(() => expect(screen.getByText(MODEL_FIXTURE.name)).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: /deploy/i })).not.toBeInTheDocument()
  })

  it('clicking Deploy creates an inference instance and navigates to /inferences', async () => {
    const created = { id: 'inst-new', model_id: MODEL_WITH_GGUF.id, model_path: 'model/abc.gguf', status: 'pending', pod_name: '', pod_namespace: 'default', idle_timeout_minutes: 120, last_used_at: null, created_at: new Date().toISOString(), updated_at: new Date().toISOString() }
    server.use(
      http.get(`${BASE}/api/models/:id`, () => HttpResponse.json(MODEL_WITH_GGUF)),
      http.post(`${BASE}/api/inferences`, () => HttpResponse.json(created, { status: 201 })),
      http.post(`${BASE}/api/inferences/:id/start`, () => HttpResponse.json({ ...created, status: 'initializing' })),
    )
    renderPage(MODEL_WITH_GGUF.id)
    await waitFor(() => screen.getByRole('button', { name: /deploy/i }))
    await userEvent.click(screen.getByRole('button', { name: /deploy/i }))
    await waitFor(() =>
      expect(screen.getByText('inferences-page')).toBeInTheDocument(),
    )
  })
})