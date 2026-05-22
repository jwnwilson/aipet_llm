import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DatasetUpload } from '@/components/DatasetUpload'

// Mock the API module so JSDOM FormData never reaches the MSW/undici stack.
// (JSDOM's FormData is incompatible with undici's Request constructor used
// internally by the MSW XHR interceptor, causing requests to hang.)
vi.mock('@/api/datasets', () => ({
  uploadTrainDataset: vi.fn(),
  uploadEvalDataset: vi.fn(),
  createDataset: vi.fn(),
  listDatasets: vi.fn().mockResolvedValue([]),
  deleteDataset: vi.fn(),
}))

import {
  uploadTrainDataset,
  uploadEvalDataset,
} from '@/api/datasets'

function renderComponent() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <DatasetUpload />
    </QueryClientProvider>
  )
}

function makeJsonlFile(name: string): File {
  return new File(['{"prompt":"a","completion":"b"}\n'], name, {
    type: 'application/octet-stream',
  })
}

beforeEach(() => {
  vi.mocked(uploadTrainDataset).mockResolvedValue({ key: 'datasets/train.jsonl' })
  vi.mocked(uploadEvalDataset).mockResolvedValue({ key: 'datasets/eval.jsonl' })
})

describe('DatasetUpload', () => {
  it('renders train and eval file inputs', () => {
    renderComponent()
    expect(screen.getByLabelText(/training dataset/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/eval dataset/i)).toBeInTheDocument()
  })

  it('renders an upload button', () => {
    renderComponent()
    expect(screen.getByRole('button', { name: /upload/i })).toBeInTheDocument()
  })

  it('shows success message after uploading both files', async () => {
    renderComponent()
    const trainInput = screen.getByLabelText(/training dataset/i)
    const evalInput = screen.getByLabelText(/eval dataset/i)
    await userEvent.upload(trainInput, makeJsonlFile('train.jsonl'))
    await userEvent.upload(evalInput, makeJsonlFile('eval.jsonl'))
    await userEvent.click(screen.getByRole('button', { name: /upload/i }))
    await waitFor(() =>
      expect(screen.getByText(/uploaded successfully/i)).toBeInTheDocument()
    )
  })

  it('re-enables button after upload completes', async () => {
    renderComponent()
    const trainInput = screen.getByLabelText(/training dataset/i)
    await userEvent.upload(trainInput, makeJsonlFile('train.jsonl'))
    await userEvent.click(screen.getByRole('button', { name: /upload/i }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /upload/i })).not.toBeDisabled()
    )
  })

  it('shows validation error when no files are selected', async () => {
    renderComponent()
    await userEvent.click(screen.getByRole('button', { name: /upload/i }))
    expect(screen.getByText(/select at least one file/i)).toBeInTheDocument()
  })

  it('shows error message with which upload failed when server returns 500', async () => {
    vi.mocked(uploadTrainDataset).mockRejectedValueOnce(
      new Error('Request failed with status code 500')
    )
    renderComponent()
    const trainInput = screen.getByLabelText(/training dataset/i)
    await userEvent.upload(trainInput, makeJsonlFile('train.jsonl'))
    await userEvent.click(screen.getByRole('button', { name: /upload/i }))
    await waitFor(() =>
      expect(screen.getByText(/training upload failed/i)).toBeInTheDocument()
    )
  })
})
