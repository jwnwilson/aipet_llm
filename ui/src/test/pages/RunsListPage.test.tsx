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

describe('RunsListPage — name display', () => {
  afterEach(() => resetHandlerState())

  it('shows workflow_id in link label when name is null', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByRole('link', { name: RUN_FIXTURE.workflow_id })).toBeInTheDocument()
    })
  })

  it('shows name in link label when name is set', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs', ({ request }) => {
        const url = new URL(request.url)
        const page = parseInt(url.searchParams.get('page') ?? '1', 10)
        const limit = parseInt(url.searchParams.get('limit') ?? '50', 10)
        const items = [{ ...RUN_FIXTURE, name: 'smoke-test-run' }, RUN_FIXTURE_2]
        return HttpResponse.json({ items, total: items.length, page, limit, pages: 1 })
      })
    )
    renderPage()
    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'smoke-test-run' })).toBeInTheDocument()
    })
  })

  it('shows workflow_id as secondary text when name is set', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs', ({ request }) => {
        const url = new URL(request.url)
        const page = parseInt(url.searchParams.get('page') ?? '1', 10)
        const limit = parseInt(url.searchParams.get('limit') ?? '50', 10)
        const items = [{ ...RUN_FIXTURE, name: 'smoke-test-run' }, RUN_FIXTURE_2]
        return HttpResponse.json({ items, total: items.length, page, limit, pages: 1 })
      })
    )
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('smoke-test-run')).toBeInTheDocument()
    })
    expect(screen.getByText(RUN_FIXTURE.workflow_id)).toBeInTheDocument()
  })

  it('does not show workflow_id as secondary text when name is null', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByRole('link', { name: RUN_FIXTURE.workflow_id })).toBeInTheDocument()
    })
    // workflow_id appears only in the primary label, not as a dimmed secondary line
    expect(screen.getAllByText(RUN_FIXTURE.workflow_id)).toHaveLength(1)
  })
})
