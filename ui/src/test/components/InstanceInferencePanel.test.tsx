import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/msw/server'
import { InstanceInferencePanel } from '@/components/InstanceInferencePanel'

const BASE = 'http://localhost:8000'

function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {children}
    </QueryClientProvider>
  )
}

describe('InstanceInferencePanel', () => {
  it('renders the run inference button', () => {
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    expect(screen.getByRole('button', { name: /run inference/i })).toBeInTheDocument()
  })

  it('renders a textarea with default payload containing pet_stats', () => {
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    const textarea = screen.getByRole('textbox') as HTMLTextAreaElement
    expect(textarea.value).toContain('pet_stats')
  })

  it('shows error on invalid JSON before submitting', async () => {
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    await userEvent.clear(screen.getByRole('textbox'))
    await userEvent.type(screen.getByRole('textbox'), 'not json')
    await userEvent.click(screen.getByRole('button', { name: /run inference/i }))
    expect(screen.getByText(/invalid json/i)).toBeInTheDocument()
  })

  it('displays inference result action on success', async () => {
    server.use(
      http.post(`${BASE}/api/inferences/inst-1/infer`, () =>
        HttpResponse.json({ action: 'SLEEP', stat: null, target_object_id: 'bed-1', confidence: 0.85 }),
      ),
    )
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    await userEvent.click(screen.getByRole('button', { name: /run inference/i }))
    await waitFor(() => expect(screen.getByText('SLEEP')).toBeInTheDocument())
  })

  it('displays error message on API failure', async () => {
    server.use(
      http.post(`${BASE}/api/inferences/inst-1/infer`, () =>
        HttpResponse.json({ detail: 'not available' }, { status: 409 }),
      ),
    )
    render(<InstanceInferencePanel instanceId="inst-1" />, { wrapper })
    await userEvent.click(screen.getByRole('button', { name: /run inference/i }))
    await waitFor(() => expect(screen.getByText(/inference failed/i)).toBeInTheDocument())
  })
})
