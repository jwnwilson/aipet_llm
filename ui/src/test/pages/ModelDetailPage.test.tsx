import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { ModelDetailPage } from '@/pages/ModelDetailPage'
import { MODEL_FIXTURE, RUN_FIXTURE } from '../msw/fixtures'
import { server } from '../msw/server'

function renderPage(modelId: string = MODEL_FIXTURE.id) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/models/${modelId}`]}>
        <Routes>
          <Route path="/models/:id" element={<ModelDetailPage />} />
          <Route path="/runs/:id" element={<div>run-detail</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('ModelDetailPage — run name input', () => {
  it('renders the run name input', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/run name \(optional\)/i)).toBeInTheDocument()
    })
  })

  it('run name input starts empty', async () => {
    renderPage()
    await waitFor(() => screen.getByPlaceholderText(/run name \(optional\)/i))
    expect(screen.getByPlaceholderText(/run name \(optional\)/i)).toHaveValue('')
  })

  it('accepts text in the run name input', async () => {
    renderPage()
    await waitFor(() => screen.getByPlaceholderText(/run name \(optional\)/i))
    await userEvent.type(screen.getByPlaceholderText(/run name \(optional\)/i), 'smoke-run')
    expect(screen.getByPlaceholderText(/run name \(optional\)/i)).toHaveValue('smoke-run')
  })

  it('sends the name to the trigger API when provided', async () => {
    let capturedBody: Record<string, unknown> | null = null
    server.use(
      http.post('http://localhost:8000/api/runs/trigger', async ({ request }) => {
        capturedBody = await request.json() as Record<string, unknown>
        return HttpResponse.json({ run_id: RUN_FIXTURE.id }, { status: 202 })
      })
    )

    renderPage()
    await waitFor(() => screen.getByPlaceholderText(/run name \(optional\)/i))
    await userEvent.type(screen.getByPlaceholderText(/run name \(optional\)/i), 'my-named-run')
    await userEvent.click(screen.getByRole('button', { name: /^run$/i }))

    await waitFor(() => expect(capturedBody).not.toBeNull())
    expect(capturedBody).toMatchObject({ model_id: MODEL_FIXTURE.id, name: 'my-named-run' })
  })

  it('sends null name when the input is empty', async () => {
    let capturedBody: Record<string, unknown> | null = null
    server.use(
      http.post('http://localhost:8000/api/runs/trigger', async ({ request }) => {
        capturedBody = await request.json() as Record<string, unknown>
        return HttpResponse.json({ run_id: RUN_FIXTURE.id }, { status: 202 })
      })
    )

    renderPage()
    await waitFor(() => screen.getByRole('button', { name: /^run$/i }))
    await userEvent.click(screen.getByRole('button', { name: /^run$/i }))

    await waitFor(() => expect(capturedBody).not.toBeNull())
    expect(capturedBody).toMatchObject({ model_id: MODEL_FIXTURE.id, name: null })
  })

  it('disables the name input while the run is being triggered', async () => {
    server.use(
      http.post('http://localhost:8000/api/runs/trigger', async () => {
        await new Promise(resolve => setTimeout(resolve, 200))
        return HttpResponse.json({ run_id: RUN_FIXTURE.id }, { status: 202 })
      })
    )

    renderPage()
    await waitFor(() => screen.getByRole('button', { name: /^run$/i }))
    const input = screen.getByPlaceholderText(/run name \(optional\)/i)
    expect(input).not.toBeDisabled()

    await userEvent.click(screen.getByRole('button', { name: /^run$/i }))
    await waitFor(() => expect(input).toBeDisabled())
  })

  it('navigates to the run detail page after triggering', async () => {
    renderPage()
    await waitFor(() => screen.getByRole('button', { name: /^run$/i }))
    await userEvent.click(screen.getByRole('button', { name: /^run$/i }))
    await waitFor(() => {
      expect(screen.getByText('run-detail')).toBeInTheDocument()
    })
  })
})
