import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DatasetsPage } from '@/pages/DatasetsPage'
import { TRAIN_DATASET_FIXTURE } from '../msw/fixtures'

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <DatasetsPage />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

function mockMobile() {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: query === '(max-width: 767px)',
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })
}

describe('DatasetsPage — desktop', () => {
  it('renders the dataset catalog table', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByRole('table')).toBeInTheDocument())
  })
})

describe('DatasetsPage — mobile', () => {
  it('renders dataset mobile cards instead of table', async () => {
    mockMobile()
    renderPage()
    await waitFor(() =>
      expect(screen.getAllByTestId('dataset-mobile-card').length).toBeGreaterThan(0)
    )
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('mobile card shows dataset name', async () => {
    mockMobile()
    renderPage()
    await waitFor(() => screen.getAllByTestId('dataset-mobile-card'))
    const cards = screen.getAllByTestId('dataset-mobile-card')
    const names = cards.map(c => c.textContent ?? '')
    expect(names.some(t => t.includes(TRAIN_DATASET_FIXTURE.name))).toBe(true)
  })

  it('mobile card shows dataset type badge', async () => {
    mockMobile()
    renderPage()
    await waitFor(() => screen.getAllByTestId('dataset-mobile-card'))
    const cards = screen.getAllByTestId('dataset-mobile-card')
    const texts = cards.map(c => c.textContent ?? '')
    expect(texts.some(t => t.includes('train'))).toBe(true)
  })
})
