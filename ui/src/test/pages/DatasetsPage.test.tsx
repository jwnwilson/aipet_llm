import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { DatasetsPage } from '@/pages/DatasetsPage'
import { TRAIN_DATASET_FIXTURE, EVAL_DATASET_FIXTURE } from '../msw/fixtures'
import { server } from '../msw/server'
import { resetHandlerState } from '../msw/handlers'
import type { Dataset } from '@/types'

// We need to mock createDataset because JSDOM's FormData is incompatible with
// undici's Request constructor used by the MSW XHR interceptor — the request
// hangs before the handler is ever reached.  GET / DELETE endpoints use JSON
// bodies and continue to run through MSW as normal.
vi.mock('@/api/datasets', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/datasets')>()
  return {
    ...actual,
    createDataset: vi.fn(),
  }
})

import { createDataset } from '@/api/datasets'

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <DatasetsPage />
    </QueryClientProvider>
  )
}

const CREATED_DATASET: Dataset = {
  id: 'ds-new-1',
  name: 'my-dataset',
  description: '',
  dataset_type: 'train',
  key: 'datasets/ds-new-1.jsonl',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
}

describe('DatasetsPage', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    resetHandlerState()
  })

  beforeEach(() => {
    vi.mocked(createDataset).mockResolvedValue(CREATED_DATASET)
  })

  describe('dataset list', () => {
    it('renders a row for each existing dataset', async () => {
      renderPage()
      await waitFor(() => {
        expect(screen.getByText(TRAIN_DATASET_FIXTURE.name)).toBeInTheDocument()
        expect(screen.getByText(EVAL_DATASET_FIXTURE.name)).toBeInTheDocument()
      })
    })

    it('shows "No datasets" message when list is empty', async () => {
      server.use(
        http.get('http://localhost:8000/api/datasets', () => HttpResponse.json([]))
      )
      renderPage()
      await waitFor(() =>
        expect(screen.getByText(/no datasets uploaded yet/i)).toBeInTheDocument()
      )
    })

    it('shows an error message when the list request fails', async () => {
      server.use(
        http.get('http://localhost:8000/api/datasets', () =>
          HttpResponse.json({ detail: 'Server error' }, { status: 500 })
        )
      )
      renderPage()
      await waitFor(() =>
        expect(screen.getByText(/failed to load datasets/i)).toBeInTheDocument()
      )
    })

    it('renders a type badge for each dataset', async () => {
      renderPage()
      await waitFor(() => {
        expect(screen.getByText('train')).toBeInTheDocument()
        expect(screen.getByText('eval')).toBeInTheDocument()
      })
    })
  })

  describe('delete', () => {
    it('removes the dataset row after confirming delete', async () => {
      vi.spyOn(window, 'confirm').mockReturnValue(true)
      renderPage()
      await waitFor(() => screen.getByText(TRAIN_DATASET_FIXTURE.name))

      const deleteBtn = screen.getByRole('button', {
        name: new RegExp(`delete dataset ${TRAIN_DATASET_FIXTURE.name}`, 'i'),
      })
      await userEvent.click(deleteBtn)

      await waitFor(() =>
        expect(screen.queryByText(TRAIN_DATASET_FIXTURE.name)).not.toBeInTheDocument()
      )
    })

    it('keeps the row when delete is cancelled', async () => {
      vi.spyOn(window, 'confirm').mockReturnValue(false)
      renderPage()
      await waitFor(() => screen.getByText(TRAIN_DATASET_FIXTURE.name))

      const deleteBtn = screen.getByRole('button', {
        name: new RegExp(`delete dataset ${TRAIN_DATASET_FIXTURE.name}`, 'i'),
      })
      await userEvent.click(deleteBtn)

      expect(screen.getByText(TRAIN_DATASET_FIXTURE.name)).toBeInTheDocument()
    })
  })

  describe('upload form', () => {
    it('renders the upload form with name, type, description, and file inputs', () => {
      renderPage()
      expect(screen.getByLabelText(/^name$/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/file/i)).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /upload dataset/i })).toBeInTheDocument()
    })

    it('shows an error when submitting without a file', async () => {
      renderPage()
      await userEvent.type(screen.getByLabelText(/^name$/i), 'new-ds')
      await userEvent.click(screen.getByRole('button', { name: /upload dataset/i }))
      await waitFor(() =>
        expect(screen.getByText(/please select a file/i)).toBeInTheDocument()
      )
    })

    it('shows an error when submitting without a name', async () => {
      renderPage()
      // Provide a file so we get past the file-required check
      const file = new File(['{}'], 'data.jsonl', { type: 'application/json' })
      await userEvent.upload(screen.getByLabelText(/file/i), file)
      await userEvent.click(screen.getByRole('button', { name: /upload dataset/i }))
      await waitFor(() =>
        expect(screen.getByText(/please enter a name/i)).toBeInTheDocument()
      )
    })

    it('shows success message and clears the form after upload', async () => {
      renderPage()
      await userEvent.type(screen.getByLabelText(/^name$/i), 'my-dataset')

      const file = new File(['{"action":"EAT"}'], 'data.jsonl', { type: 'application/json' })
      await userEvent.upload(screen.getByLabelText(/file/i), file)

      await userEvent.click(screen.getByRole('button', { name: /upload dataset/i }))

      await waitFor(() =>
        expect(screen.getByText(/dataset uploaded successfully/i)).toBeInTheDocument()
      )
      expect(screen.getByLabelText(/^name$/i)).toHaveValue('')
    })
  })
})
