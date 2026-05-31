import { afterEach, describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { RunsListPage } from '@/pages/RunsListPage'
import { MODEL_FIXTURE, MODEL_FIXTURE_2, RUN_FIXTURE, RUN_FIXTURE_2 } from '../msw/fixtures'
import { resetHandlerState } from '../msw/handlers'
import { server } from '../msw/server'

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <RunsListPage />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('RunsListPage', () => {
  afterEach(() => resetHandlerState())

  it('renders a section heading for each model', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText(MODEL_FIXTURE.name)).toBeInTheDocument()
      expect(screen.getByText(MODEL_FIXTURE_2.name)).toBeInTheDocument()
    })
  })

  it('shows run count label under each model heading', async () => {
    renderPage()
    await waitFor(() => {
      const labels = screen.getAllByText(/1 run$/)
      expect(labels.length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders a link to each run detail page', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByRole('link', { name: RUN_FIXTURE.workflow_id })).toBeInTheDocument()
      expect(screen.getByRole('link', { name: RUN_FIXTURE_2.workflow_id })).toBeInTheDocument()
    })
  })

  it('shows the empty state when there are no runs', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs', () =>
        HttpResponse.json({ items: [], total: 0, page: 1, limit: 50, pages: 1 })
      )
    )
    renderPage()
    await waitFor(() => expect(screen.getByText(/no runs yet/i)).toBeInTheDocument())
  })

  it('falls back to "Unknown model" for a run with an unrecognised model_id', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs', () =>
        HttpResponse.json({
          items: [{ ...RUN_FIXTURE, model_id: 'orphan-model-id' }],
          total: 1, page: 1, limit: 50, pages: 1,
        })
      )
    )
    renderPage()
    await waitFor(() => expect(screen.getByText(/unknown model/i)).toBeInTheDocument())
  })
})
