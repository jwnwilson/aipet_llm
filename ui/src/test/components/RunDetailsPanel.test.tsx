import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { RunDetailsPanel } from '@/components/RunDetailsPanel'
import { RUN_FIXTURE, TEMPORAL_DETAILS_FIXTURE, RUN_LOGS_FIXTURE } from '../msw/fixtures'
import { server } from '../msw/server'

function renderPanel(runOverride: Partial<typeof RUN_FIXTURE> = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const run = { ...RUN_FIXTURE, ...runOverride }
  render(
    <QueryClientProvider client={client}>
      <RunDetailsPanel runId={run.id} run={run} />
    </QueryClientProvider>
  )
}

describe('RunDetailsPanel — collapsed state', () => {
  it('renders the toggle button with "Stage details" label', () => {
    renderPanel()
    expect(screen.getByRole('button', { name: /stage details/i })).toBeInTheDocument()
  })

  it('does not show temporal or log content when collapsed', () => {
    renderPanel()
    expect(screen.queryByText(/workflow id/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/no logs captured/i)).not.toBeInTheDocument()
  })

  it('does not fetch from the network when collapsed', () => {
    let temporalCalled = false
    server.use(
      http.get('http://localhost:8000/api/runs/:id/temporal', () => {
        temporalCalled = true
        return HttpResponse.json(TEMPORAL_DETAILS_FIXTURE)
      })
    )
    renderPanel()
    expect(temporalCalled).toBe(false)
  })
})

describe('RunDetailsPanel — expanded state', () => {
  it('shows temporal workflow details after toggle', async () => {
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    await waitFor(() =>
      expect(screen.getByText(TEMPORAL_DETAILS_FIXTURE.workflow_id)).toBeInTheDocument()
    )
    expect(screen.getByText(TEMPORAL_DETAILS_FIXTURE.temporal_run_id)).toBeInTheDocument()
    expect(screen.getByText('RUNNING')).toBeInTheDocument()
  })

  it('shows log content after toggle when logs exist', async () => {
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    await waitFor(() =>
      expect(screen.getByText(/epoch 1\/3/)).toBeInTheDocument()
    )
  })

  it('shows "No logs captured" when logs are null', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs/:id/logs', ({ params }) => {
        if (params.id === RUN_FIXTURE.id)
          return HttpResponse.json({ logs: null, source: null })
        return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
      })
    )
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    await waitFor(() =>
      expect(screen.getByText(/no logs captured/i)).toBeInTheDocument()
    )
  })

  it('collapses again on second toggle', async () => {
    renderPanel()
    const button = screen.getByRole('button', { name: /stage details/i })
    await userEvent.click(button)
    await waitFor(() =>
      expect(screen.getByText(TEMPORAL_DETAILS_FIXTURE.workflow_id)).toBeInTheDocument()
    )
    await userEvent.click(button)
    expect(screen.queryByText(TEMPORAL_DETAILS_FIXTURE.workflow_id)).not.toBeInTheDocument()
  })

  it('shows error message when temporal fetch fails', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs/:id/temporal', () =>
        HttpResponse.json({ detail: 'Temporal unreachable' }, { status: 502 })
      )
    )
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    await waitFor(() =>
      expect(screen.getByText(/failed to load workflow details/i)).toBeInTheDocument()
    )
  })
})

describe('RunDetailsPanel — accessibility', () => {
  it('has aria-expanded=false on initial render', () => {
    renderPanel()
    expect(screen.getByRole('button', { name: /stage details/i })).toHaveAttribute(
      'aria-expanded',
      'false'
    )
  })

  it('sets aria-expanded=true after expanding', async () => {
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    expect(screen.getByRole('button', { name: /stage details/i })).toHaveAttribute(
      'aria-expanded',
      'true'
    )
  })

  it('sets aria-expanded back to false after collapsing', async () => {
    renderPanel()
    const button = screen.getByRole('button', { name: /stage details/i })
    await userEvent.click(button)
    await userEvent.click(button)
    expect(button).toHaveAttribute('aria-expanded', 'false')
  })
})

describe('RunDetailsPanel — temporal data display', () => {
  it('renders the "Started" row when start_time is present', async () => {
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    await waitFor(() =>
      expect(screen.getByText(TEMPORAL_DETAILS_FIXTURE.workflow_id)).toBeInTheDocument()
    )
    expect(screen.getByText(/started/i)).toBeInTheDocument()
  })

  it('does not render "Finished" row when close_time is null', async () => {
    // TEMPORAL_DETAILS_FIXTURE has close_time: null
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    await waitFor(() =>
      expect(screen.getByText(TEMPORAL_DETAILS_FIXTURE.workflow_id)).toBeInTheDocument()
    )
    expect(screen.queryByText(/finished/i)).not.toBeInTheDocument()
  })

  it('renders "Finished" row when close_time is set', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs/:id/temporal', ({ params }) => {
        if (params.id === RUN_FIXTURE.id)
          return HttpResponse.json({
            ...TEMPORAL_DETAILS_FIXTURE,
            status: 'COMPLETED',
            close_time: '2024-01-02T12:00:00.000Z',
          })
        return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
      })
    )
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    await waitFor(() => expect(screen.getByText(/finished/i)).toBeInTheDocument())
  })
})

describe('RunDetailsPanel — loading state', () => {
  it('shows only the toggle button while data is loading', async () => {
    // Delay responses so we can inspect the loading state
    server.use(
      http.get('http://localhost:8000/api/runs/:id/temporal', async () => {
        await new Promise(resolve => setTimeout(resolve, 5000))
        return HttpResponse.json(TEMPORAL_DETAILS_FIXTURE)
      }),
      http.get('http://localhost:8000/api/runs/:id/logs', async () => {
        await new Promise(resolve => setTimeout(resolve, 5000))
        return HttpResponse.json(RUN_LOGS_FIXTURE)
      })
    )
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    // Queries are in-flight — no content yet
    expect(screen.queryByText(TEMPORAL_DETAILS_FIXTURE.workflow_id)).not.toBeInTheDocument()
    expect(screen.queryByText(/epoch 1\/3/)).not.toBeInTheDocument()
    expect(screen.queryByText(/no logs captured/i)).not.toBeInTheDocument()
    // Button is still accessible
    expect(screen.getByRole('button', { name: /stage details/i })).toBeInTheDocument()
  })
})

describe('RunDetailsPanel — logs error handling', () => {
  it('shows no log error UI when the logs endpoint returns 500', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs/:id/logs', () =>
        HttpResponse.json({ detail: 'Internal server error' }, { status: 500 })
      )
    )
    renderPanel()
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    // Temporal data still loads fine
    await waitFor(() =>
      expect(screen.getByText(TEMPORAL_DETAILS_FIXTURE.workflow_id)).toBeInTheDocument()
    )
    // No logs-specific error message shown
    expect(screen.queryByText(/failed to load.*log/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/no logs captured/i)).not.toBeInTheDocument()
  })
})

describe('RunDetailsPanel — polling', () => {
  it('does not refetch after initial load for a completed run', async () => {
    let callCount = 0
    server.use(
      http.get('http://localhost:8000/api/runs/:id/temporal', ({ params }) => {
        if (params.id === RUN_FIXTURE.id) {
          callCount++
          return HttpResponse.json({ ...TEMPORAL_DETAILS_FIXTURE, status: 'COMPLETED' })
        }
        return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
      })
    )
    // Render with a completed run — isRunActive returns false → refetchInterval: false
    renderPanel({ status: 'completed' })
    await userEvent.click(screen.getByRole('button', { name: /stage details/i }))
    await waitFor(() => expect(callCount).toBe(1))
    // Count stays at 1 — no background polling
    await new Promise(resolve => setTimeout(resolve, 200))
    expect(callCount).toBe(1)
  })
})
