import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DatasetUpload } from '@/components/DatasetUpload'

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
})
