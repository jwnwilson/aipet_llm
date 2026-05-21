import { afterEach, describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { LinkedDatasetsCard } from '@/components/LinkedDatasetsCard'
import { MODEL_FIXTURE, TRAIN_DATASET_FIXTURE, EVAL_DATASET_FIXTURE } from '../msw/fixtures'
import { server } from '../msw/server'
import { resetHandlerState } from '../msw/handlers'

function renderCard(modelOverride = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  const model = { ...MODEL_FIXTURE, ...modelOverride }
  render(
    <QueryClientProvider client={client}>
      <LinkedDatasetsCard model={model} />
    </QueryClientProvider>
  )
}

describe('LinkedDatasetsCard', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    resetHandlerState()
  })

  it('shows "Not linked" when model train_data does not match any dataset key', async () => {
    renderCard()
    await waitFor(() =>
      expect(screen.getAllByText(/not linked/i).length).toBeGreaterThanOrEqual(1)
    )
  })

  it('shows the linked train dataset name when train_data matches a dataset key', async () => {
    renderCard({ train_data: TRAIN_DATASET_FIXTURE.key })
    await waitFor(() =>
      expect(screen.getByText(TRAIN_DATASET_FIXTURE.name)).toBeInTheDocument()
    )
  })

  it('shows the linked eval dataset name when eval_data matches a dataset key', async () => {
    renderCard({ eval_data: EVAL_DATASET_FIXTURE.key })
    await waitFor(() =>
      expect(screen.getByText(EVAL_DATASET_FIXTURE.name)).toBeInTheDocument()
    )
  })

  it('Save button is disabled when no selection has changed', async () => {
    renderCard()
    await waitFor(() => screen.getByRole('button', { name: /save/i }))
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()
  })

  it('Save button becomes enabled after selecting a train dataset', async () => {
    renderCard()
    await userEvent.click(await screen.findByRole('combobox', { name: /select training dataset/i }))
    await userEvent.click(await screen.findByText(TRAIN_DATASET_FIXTURE.name))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /save/i })).not.toBeDisabled()
    )
  })

  it('calls updateModel with new train_data key after saving', async () => {
    let capturedBody: unknown
    server.use(
      http.put('http://localhost:8000/api/models/:id', async ({ request }) => {
        capturedBody = await request.json()
        return HttpResponse.json({ ...MODEL_FIXTURE, train_data: TRAIN_DATASET_FIXTURE.key })
      })
    )

    renderCard()
    await userEvent.click(await screen.findByRole('combobox', { name: /select training dataset/i }))
    await userEvent.click(await screen.findByText(TRAIN_DATASET_FIXTURE.name))
    await userEvent.click(screen.getByRole('button', { name: /save/i }))

    await waitFor(() =>
      expect((capturedBody as Record<string, unknown>)?.train_data).toBe(TRAIN_DATASET_FIXTURE.key)
    )
  })

  it('shows error message when save fails', async () => {
    server.use(
      http.put('http://localhost:8000/api/models/:id', () =>
        HttpResponse.json({ detail: 'Server error' }, { status: 500 })
      )
    )

    renderCard()
    await userEvent.click(await screen.findByRole('combobox', { name: /select training dataset/i }))
    await userEvent.click(await screen.findByText(TRAIN_DATASET_FIXTURE.name))
    await userEvent.click(screen.getByRole('button', { name: /save/i }))

    await waitFor(() =>
      expect(screen.getByText(/failed to save/i)).toBeInTheDocument()
    )
  })
})
