import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('@auth0/auth0-react', () => ({
  useAuth0: () => ({
    isAuthenticated: true,
    isLoading: false,
    user: { email: 'test@example.com', 'https://aipet/roles': [] },
    loginWithRedirect: vi.fn(),
    logout: vi.fn(),
  }),
}))

vi.mock('@/components/TokenSync', () => ({ TokenSync: () => null }))
vi.mock('@/pages/ModelsListPage', () => ({ ModelsListPage: () => <div>models</div> }))
vi.mock('@/pages/ModelFormPage', () => ({ ModelFormPage: () => <div>form</div> }))
vi.mock('@/pages/ModelDetailPage', () => ({ ModelDetailPage: () => <div>detail</div> }))
vi.mock('@/pages/RunsListPage', () => ({ RunsListPage: () => <div>runs</div> }))
vi.mock('@/pages/RunDetailPage', () => ({ RunDetailPage: () => <div>run detail</div> }))
vi.mock('@/pages/DatasetsPage', () => ({ DatasetsPage: () => <div>datasets</div> }))
vi.mock('@/pages/InferencePage', () => ({ InferencePage: () => <div>inference</div> }))
vi.mock('@/pages/UsersPage', () => ({ UsersPage: () => <div>users</div> }))

import App from '@/App'

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/models']}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('Nav', () => {
  it('renders hamburger button', () => {
    renderApp()
    expect(screen.getByRole('button', { name: /open menu/i })).toBeInTheDocument()
  })

  it('mobile menu is hidden by default', () => {
    renderApp()
    expect(screen.queryByRole('navigation', { name: /mobile navigation/i })).not.toBeInTheDocument()
  })

  it('opens mobile menu when hamburger is clicked', async () => {
    renderApp()
    await userEvent.click(screen.getByRole('button', { name: /open menu/i }))
    expect(screen.getByRole('navigation', { name: /mobile navigation/i })).toBeInTheDocument()
  })

  it('closes mobile menu when hamburger is clicked again', async () => {
    renderApp()
    await userEvent.click(screen.getByRole('button', { name: /open menu/i }))
    await userEvent.click(screen.getByRole('button', { name: /close menu/i }))
    expect(screen.queryByRole('navigation', { name: /mobile navigation/i })).not.toBeInTheDocument()
  })

  it('mobile menu contains all nav links', async () => {
    renderApp()
    await userEvent.click(screen.getByRole('button', { name: /open menu/i }))
    const mobileNav = screen.getByRole('navigation', { name: /mobile navigation/i })
    expect(mobileNav).toHaveTextContent('Models')
    expect(mobileNav).toHaveTextContent('Datasets')
    expect(mobileNav).toHaveTextContent('Runs')
    expect(mobileNav).toHaveTextContent('Inference')
  })
})
