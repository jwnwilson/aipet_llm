# UI Mobile Responsive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every page in the llm_api UI usable on mobile screens (≥320px), with a hamburger nav, responsive tables that switch to card lists, and a compact pipeline stepper.

**Architecture:** Add a `useMediaQuery` hook for JS-driven breakpoint detection; use conditional rendering (not CSS-only `hidden`) so tests can verify mobile/desktop views; update the Nav component with a hamburger drawer; swap `<table>` layouts for card lists at `max-width: 767px`.

**Tech Stack:** React 18, TypeScript, Tailwind CSS v4, Vitest + Testing Library, MSW for API mocks.

---

## File Map

| File | Change |
|------|--------|
| `ui/src/test/setup.ts` | Add `window.matchMedia` mock |
| `ui/src/hooks/useMediaQuery.ts` | New — JS breakpoint hook |
| `ui/src/test/hooks/useMediaQuery.test.ts` | New — hook tests |
| `ui/src/App.tsx` | Nav: add hamburger + mobile drawer |
| `ui/src/test/App.test.tsx` | New — Nav hamburger tests |
| `ui/src/index.css` | `.ed-page` responsive padding; body `overflow-x: hidden` |
| `ui/src/pages/ModelsListPage.tsx` | Add `ModelMobileCard`; switch to card list on mobile |
| `ui/src/test/pages/ModelsListPage.test.tsx` | Add mobile card rendering tests |
| `ui/src/pages/DatasetsPage.tsx` | Add `DatasetMobileCard`; switch to card list on mobile |
| `ui/src/test/pages/DatasetsPage.test.tsx` | New — dataset mobile card tests |
| `ui/src/components/PipelineStages.tsx` | Wrap to 2×2 grid on mobile |
| `ui/src/test/components/PipelineStages.test.tsx` | Add mobile layout test |

---

## Task 1: Add `window.matchMedia` mock and `useMediaQuery` hook

**Files:**
- Modify: `ui/src/test/setup.ts`
- Create: `ui/src/hooks/useMediaQuery.ts`
- Create: `ui/src/test/hooks/useMediaQuery.test.ts`

- [ ] **Step 1: Write failing tests for `useMediaQuery`**

Create `ui/src/test/hooks/useMediaQuery.test.ts`:

```typescript
import { renderHook } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useMediaQuery } from '@/hooks/useMediaQuery'

function mockMatchMedia(matches: boolean) {
  const listeners: Array<(e: Partial<MediaQueryListEvent>) => void> = []
  const mql = {
    matches,
    media: '',
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn((_: string, cb: (e: Partial<MediaQueryListEvent>) => void) => {
      listeners.push(cb)
    }),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
    _listeners: listeners,
  }
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockReturnValue(mql),
  })
  return mql
}

describe('useMediaQuery', () => {
  beforeEach(() => {
    mockMatchMedia(false)
  })

  it('returns false when media query does not match', () => {
    mockMatchMedia(false)
    const { result } = renderHook(() => useMediaQuery('(max-width: 767px)'))
    expect(result.current).toBe(false)
  })

  it('returns true when media query matches', () => {
    mockMatchMedia(true)
    const { result } = renderHook(() => useMediaQuery('(max-width: 767px)'))
    expect(result.current).toBe(true)
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ui && npx vitest run src/test/hooks/useMediaQuery.test.ts
```

Expected: FAIL — `Cannot find module '@/hooks/useMediaQuery'`

- [ ] **Step 3: Add `window.matchMedia` mock to global test setup**

Edit `ui/src/test/setup.ts`. Merge `vi` into the existing vitest import:

```typescript
import { afterAll, afterEach, beforeAll, vi } from 'vitest'
```

Add this block after `Element.prototype.hasPointerCapture ??= () => false`, before `beforeAll`:

```typescript
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})
```

- [ ] **Step 4: Create the `useMediaQuery` hook**

Create `ui/src/hooks/useMediaQuery.ts`:

```typescript
import { useState, useEffect } from 'react'

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia(query).matches
  })

  useEffect(() => {
    if (typeof window === 'undefined') return
    const mql = window.matchMedia(query)
    const handler = (e: MediaQueryListEvent) => setMatches(e.matches)
    mql.addEventListener('change', handler)
    return () => mql.removeEventListener('change', handler)
  }, [query])

  return matches
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ui && npx vitest run src/test/hooks/useMediaQuery.test.ts
```

Expected: 2 tests PASS

- [ ] **Step 6: Run full suite to verify no regressions**

```bash
cd ui && npm test -- --run
```

Expected: all tests pass (185+)

- [ ] **Step 7: Commit**

```bash
git add ui/src/test/setup.ts ui/src/hooks/useMediaQuery.ts ui/src/test/hooks/useMediaQuery.test.ts
git commit -m "feat: add useMediaQuery hook with matchMedia test setup"
```

---

## Task 2: Responsive navigation with hamburger menu

**Files:**
- Modify: `ui/src/App.tsx`
- Create: `ui/src/test/App.test.tsx`

- [ ] **Step 1: Write failing tests for the hamburger nav**

Create `ui/src/test/App.test.tsx`:

```typescript
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ui && npx vitest run src/test/App.test.tsx
```

Expected: FAIL — hamburger button not found

- [ ] **Step 3: Update `App.tsx` Nav component**

Replace the `Nav` function in `ui/src/App.tsx` (lines 43–74) with:

```tsx
function Nav() {
  const isAdmin = useIsAdmin()
  const [menuOpen, setMenuOpen] = useState(false)

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    [
      "font-['Outfit'] text-[0.78rem] font-medium uppercase tracking-[0.12em]",
      'pb-1 transition-colors duration-150',
      isActive
        ? 'text-[#1a1a1a] border-b-[1.5px] border-[#1a1a1a]'
        : 'text-[#3a3a36] hover:text-[#1a1a1a] border-b-[1.5px] border-transparent',
    ].join(' ')

  const mobileLinkClass = ({ isActive }: { isActive: boolean }) =>
    [
      "font-['Outfit'] text-[0.88rem] font-medium uppercase tracking-[0.12em]",
      'py-3 px-6 block w-full border-b border-[#e5e3d8] transition-colors duration-150',
      isActive ? 'text-[#1a1a1a] bg-[#f3f2ec]' : 'text-[#3a3a36] hover:text-[#1a1a1a]',
    ].join(' ')

  return (
    <header className="sticky top-0 z-40 bg-[#fafaf7]/95 backdrop-blur-sm border-b-2 border-[#1a1a1a]">
      <div className="max-w-[1240px] mx-auto px-4 sm:px-8 h-16 flex items-center gap-4 sm:gap-10">
        <Link to="/models" className="flex items-baseline gap-1 select-none shrink-0">
          <span className="font-['DM_Serif_Display'] text-[1.55rem] leading-none text-[#1a1a1a]">
            LLM
          </span>
          <span className="font-['IBM_Plex_Mono'] text-[0.85rem] text-[#888888]">.api</span>
        </Link>
        <nav className="hidden md:flex items-center gap-7">
          <NavLink to="/models" className={linkClass}>Models</NavLink>
          <NavLink to="/datasets" className={linkClass}>Datasets</NavLink>
          <NavLink to="/runs" className={linkClass}>Runs</NavLink>
          <NavLink to="/inferences" className={linkClass}>Inference</NavLink>
          {isAdmin && <NavLink to="/admin/users" className={linkClass}>Users</NavLink>}
        </nav>
        <div className="flex items-center gap-3 ml-auto">
          <AuthCluster />
          <button
            onClick={() => setMenuOpen(prev => !prev)}
            className="md:hidden flex items-center justify-center w-8 h-8 text-[#1a1a1a] shrink-0"
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={menuOpen}
          >
            {menuOpen ? (
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6 6 18M6 6l12 12" strokeLinecap="round" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 6h16M4 12h16M4 18h16" strokeLinecap="round" />
              </svg>
            )}
          </button>
        </div>
      </div>
      {menuOpen && (
        <nav
          className="md:hidden border-t border-[#d0d0c8] bg-[#fafaf7]"
          aria-label="Mobile navigation"
        >
          <NavLink to="/models" className={mobileLinkClass} onClick={() => setMenuOpen(false)}>Models</NavLink>
          <NavLink to="/datasets" className={mobileLinkClass} onClick={() => setMenuOpen(false)}>Datasets</NavLink>
          <NavLink to="/runs" className={mobileLinkClass} onClick={() => setMenuOpen(false)}>Runs</NavLink>
          <NavLink to="/inferences" className={mobileLinkClass} onClick={() => setMenuOpen(false)}>Inference</NavLink>
          {isAdmin && (
            <NavLink to="/admin/users" className={mobileLinkClass} onClick={() => setMenuOpen(false)}>Users</NavLink>
          )}
        </nav>
      )}
    </header>
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ui && npx vitest run src/test/App.test.tsx
```

Expected: 5 tests PASS

- [ ] **Step 5: Run full suite**

```bash
cd ui && npm test -- --run
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add ui/src/App.tsx ui/src/test/App.test.tsx
git commit -m "feat: add hamburger navigation menu for mobile screens"
```

---

## Task 3: Responsive page container padding

**Files:**
- Modify: `ui/src/index.css`

No new tests needed — this is a CSS-only change. The existing page tests remain the guard against regressions.

- [ ] **Step 1: Update `.ed-page` in `ui/src/index.css`**

Replace the `.ed-page` rule (lines 139–143):

```css
/* Page container — content max width */
.ed-page {
  max-width: 1240px;
  margin: 0 auto;
  padding: 2.5rem 2rem 4rem;
}
```

With:

```css
/* Page container — content max width */
.ed-page {
  max-width: 1240px;
  margin: 0 auto;
  padding: 1.25rem 1rem 3rem;
}

@media (min-width: 640px) {
  .ed-page {
    padding: 2rem 1.5rem 4rem;
  }
}

@media (min-width: 768px) {
  .ed-page {
    padding: 2.5rem 2rem 4rem;
  }
}
```

- [ ] **Step 2: Add `overflow-x: hidden` to body**

In the `body` rule in `ui/src/index.css`, add after `letter-spacing: 0.005em;`:

```css
  overflow-x: hidden;
```

- [ ] **Step 3: Run full suite to verify no regressions**

```bash
cd ui && npm test -- --run
```

Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add ui/src/index.css
git commit -m "fix: responsive page padding and prevent mobile horizontal overflow"
```

---

## Task 4: Models page mobile card view

**Files:**
- Modify: `ui/src/pages/ModelsListPage.tsx`
- Modify: `ui/src/test/pages/ModelsListPage.test.tsx`

The table is replaced with a card list when the screen width is ≤767px. Conditional rendering (not CSS `hidden`) is used so tests can assert which view is active.

- [ ] **Step 1: Write failing tests for mobile card view**

Append to `ui/src/test/pages/ModelsListPage.test.tsx` (after existing tests):

```typescript
import { vi } from 'vitest'

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

describe('ModelsListPage — mobile view', () => {
  it('renders model name as a card on mobile', async () => {
    mockMobile()
    renderPage()
    await waitFor(() =>
      expect(screen.getByTestId('model-mobile-card')).toBeInTheDocument()
    )
    expect(screen.getByTestId('model-mobile-card')).toHaveTextContent(MODEL_FIXTURE.name)
  })

  it('does not render the table on mobile', async () => {
    mockMobile()
    renderPage()
    await waitFor(() => screen.getByTestId('model-mobile-card'))
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('renders Run button in mobile card', async () => {
    mockMobile()
    renderPage()
    await waitFor(() => screen.getByTestId('model-mobile-card'))
    expect(
      screen.getByRole('button', { name: new RegExp(`trigger run for ${MODEL_FIXTURE.name}`, 'i') })
    ).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ui && npx vitest run src/test/pages/ModelsListPage.test.tsx
```

Expected: FAIL — `model-mobile-card` not found

- [ ] **Step 3: Add `useMediaQuery` import to `ModelsListPage.tsx`**

Add at the top with other imports:

```typescript
import { useMediaQuery } from '@/hooks/useMediaQuery'
```

- [ ] **Step 4: Add `ModelMobileCard` component to `ModelsListPage.tsx`**

Place after the `EmptyState` component, before `ModelsListPage`:

```tsx
function ModelMobileCard({
  model,
  index,
  onRun,
  onDelete,
  deletePending,
}: {
  model: TrainingModel
  index: number
  onRun: () => void
  onDelete: () => void
  deletePending: boolean
}) {
  const navigate = useNavigate()
  return (
    <div
      data-testid="model-mobile-card"
      className="border-b border-[#e5e3d8] px-4 py-4 bg-white last:border-b-0 cursor-pointer"
      onClick={() => navigate(`/models/${model.id}`)}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#888888]">
              {String(index + 1).padStart(2, '0')}
            </span>
            <span className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#888888]">
              {model.base_model}
            </span>
          </div>
          <h3 className="font-['DM_Serif_Display'] text-[1.05rem] text-[#1a1a1a] leading-tight">
            {model.name}
          </h3>
          {model.description && (
            <p className="font-['Outfit'] text-[0.78rem] text-[#888888] mt-0.5 line-clamp-1">
              {model.description}
            </p>
          )}
          <div className="flex items-center gap-3 mt-2 flex-wrap">
            <span className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#3a3a36]">
              {model.remote_backend}
            </span>
            <span className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#888888]">
              {model.epochs} epochs
            </span>
            {model.is_active && <RunStatusBadge status="completed" />}
          </div>
        </div>
        <div
          className="flex flex-col gap-1.5 shrink-0"
          onClick={e => e.stopPropagation()}
        >
          <Button size="sm" onClick={onRun} aria-label={`Trigger run for ${model.name}`}>
            <Play className="h-3 w-3" />Run
          </Button>
          <Button size="sm" variant="outline" asChild>
            <Link to={`/models/${model.id}/edit`} aria-label={`Edit ${model.name}`}>
              <Pencil className="h-3 w-3" />Edit
            </Link>
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={onDelete}
            disabled={deletePending}
            aria-label={`Delete ${model.name}`}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Add `isMobile` and conditional rendering in `ModelsListPage`**

Inside `ModelsListPage`, after the `filtered` const, add:

```typescript
const isMobile = useMediaQuery('(max-width: 767px)')
```

Replace the entire `<div className="bg-white border border-[#d0d0c8] ...overflow-hidden">` block that wraps the `<table>` with:

```tsx
{isMobile ? (
  <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
    {filtered.length === 0 ? (
      <div className="px-4 py-10 text-center">
        <span className="font-['DM_Serif_Display'] italic text-[#888888]">
          No models match "{search}"
        </span>
      </div>
    ) : (
      filtered.map((model, i) => (
        <ModelMobileCard
          key={model.id}
          model={model}
          index={i}
          onRun={() => setRunTarget(model)}
          onDelete={() => deleteMutation.mutate(model.id)}
          deletePending={
            deleteMutation.isPending && deleteMutation.variables === model.id
          }
        />
      ))
    )}
  </div>
) : (
  <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
    <table className="ed-table">
      <thead>
        <tr>
          <th style={{ width: '4rem' }}>№</th>
          <th>Model</th>
          <th>Base</th>
          <th>Backend</th>
          <th style={{ width: '5rem' }}>Epochs</th>
          <th>Status</th>
          <th style={{ width: '14rem' }}></th>
        </tr>
      </thead>
      <tbody>
        {filtered.length === 0 ? (
          <tr>
            <td colSpan={7} className="text-center py-10">
              <span className="font-['DM_Serif_Display'] italic text-[#888888]">
                No models match "{search}"
              </span>
            </td>
          </tr>
        ) : (
          filtered.map((model, i) => (
            <tr
              key={model.id}
              className="cursor-pointer"
              onClick={() => navigate(`/models/${model.id}`)}
            >
              <td>
                <span className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#888888]">
                  {String(i + 1).padStart(2, '0')}
                </span>
              </td>
              <td>
                <div className="font-['DM_Serif_Display'] text-[1.05rem] text-[#1a1a1a] leading-tight">
                  {model.name}
                </div>
                {model.description && (
                  <div className="font-['Outfit'] text-[0.78rem] text-[#888888] mt-0.5 line-clamp-1 max-w-md">
                    {model.description}
                  </div>
                )}
              </td>
              <td className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#3a3a36]">
                {model.base_model}
              </td>
              <td className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#3a3a36]">
                {model.remote_backend}
              </td>
              <td className="font-['IBM_Plex_Mono'] text-[0.85rem] text-[#1a1a1a]">
                {model.epochs}
              </td>
              <td>
                {model.is_active ? (
                  <RunStatusBadge status="completed" />
                ) : (
                  <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#b3b1a6]">
                    —
                  </span>
                )}
              </td>
              <td onClick={e => e.stopPropagation()}>
                <div className="flex gap-2 justify-end">
                  <Button
                    size="sm"
                    onClick={() => setRunTarget(model)}
                    aria-label={`Trigger run for ${model.name}`}
                  >
                    <Play className="h-3 w-3" />Run
                  </Button>
                  <Button size="sm" variant="outline" asChild>
                    <Link
                      to={`/models/${model.id}/edit`}
                      aria-label={`Edit ${model.name}`}
                    >
                      <Pencil className="h-3 w-3" />Edit
                    </Link>
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => deleteMutation.mutate(model.id)}
                    disabled={
                      deleteMutation.isPending &&
                      deleteMutation.variables === model.id
                    }
                    aria-label={`Delete ${model.name}`}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </td>
            </tr>
          ))
        )}
      </tbody>
    </table>
  </div>
)}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd ui && npx vitest run src/test/pages/ModelsListPage.test.tsx
```

Expected: 8 tests PASS (existing 5 + new 3)

- [ ] **Step 7: Run full suite**

```bash
cd ui && npm test -- --run
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add ui/src/pages/ModelsListPage.tsx ui/src/test/pages/ModelsListPage.test.tsx
git commit -m "feat: mobile card view for models list page"
```

---

## Task 5: Datasets page mobile card view

**Files:**
- Modify: `ui/src/pages/DatasetsPage.tsx`
- Create: `ui/src/test/pages/DatasetsPage.test.tsx`

- [ ] **Step 1: Write failing tests**

Create `ui/src/test/pages/DatasetsPage.test.tsx`:

```typescript
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
      expect(screen.getByTestId('dataset-mobile-card')).toBeInTheDocument()
    )
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('mobile card shows dataset name', async () => {
    mockMobile()
    renderPage()
    await waitFor(() => screen.getByTestId('dataset-mobile-card'))
    expect(screen.getByTestId('dataset-mobile-card')).toHaveTextContent(
      TRAIN_DATASET_FIXTURE.name
    )
  })

  it('mobile card shows dataset type badge', async () => {
    mockMobile()
    renderPage()
    await waitFor(() => screen.getByTestId('dataset-mobile-card'))
    expect(screen.getByTestId('dataset-mobile-card')).toHaveTextContent('train')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ui && npx vitest run src/test/pages/DatasetsPage.test.tsx
```

Expected: FAIL — `dataset-mobile-card` not found

- [ ] **Step 3: Add `useMediaQuery` import to `DatasetsPage.tsx`**

```typescript
import { useMediaQuery } from '@/hooks/useMediaQuery'
```

- [ ] **Step 4: Add `DatasetMobileCard` component to `DatasetsPage.tsx`**

Place after `DatasetRow`, before `UploadDropzone`:

```tsx
function DatasetMobileCard({
  dataset,
  onDelete,
  index,
}: {
  dataset: Dataset
  onDelete: (id: string) => void
  index: number
}) {
  return (
    <div
      data-testid="dataset-mobile-card"
      className="border-b border-[#e5e3d8] px-4 py-4 bg-white last:border-b-0"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#888888]">
              {String(index + 1).padStart(2, '0')}
            </span>
            <span
              className={[
                "inline-flex items-center font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em]",
                'px-2 py-[3px] rounded-[2px] border',
                dataset.dataset_type === 'train'
                  ? 'border-[#1a1a1a] bg-[#1a1a1a] text-[#fafaf7]'
                  : 'border-[#2d6a4f] bg-[#e8efe9] text-[#2d6a4f]',
              ].join(' ')}
            >
              {dataset.dataset_type}
            </span>
          </div>
          <p className="font-['IBM_Plex_Mono'] text-[0.88rem] text-[#1a1a1a]">
            {dataset.name}
          </p>
          {dataset.description && (
            <p className="font-['Outfit'] text-[0.78rem] text-[#888888] mt-0.5 line-clamp-2">
              {dataset.description}
            </p>
          )}
          <p className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#888888] mt-1 truncate">
            {dataset.key}
          </p>
          <p className="font-['Outfit'] text-[0.72rem] text-[#888888] mt-0.5">
            {new Date(dataset.created_at).toLocaleDateString()}
          </p>
        </div>
        <button
          onClick={() => onDelete(dataset.id)}
          aria-label={`Delete dataset ${dataset.name}`}
          className="text-[#888888] hover:text-[#7f1d1d] transition-colors p-1.5 shrink-0 mt-1"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Add `isMobile` and conditional rendering in `DatasetsPage`**

In `DatasetsPage`, add after `deleteMutation`:

```typescript
const isMobile = useMediaQuery('(max-width: 767px)')
```

Replace the `<table className="ed-table">` block in the catalog section with:

```tsx
{isMobile ? (
  <div>
    {datasets.map((ds, i) => (
      <DatasetMobileCard key={ds.id} dataset={ds} onDelete={handleDelete} index={i} />
    ))}
  </div>
) : (
  <table className="ed-table">
    <thead>
      <tr>
        <th style={{ width: '3rem' }}>№</th>
        <th>Name</th>
        <th>Type</th>
        <th>Description</th>
        <th>Storage key</th>
        <th>Created</th>
        <th style={{ width: '3rem' }}></th>
      </tr>
    </thead>
    <tbody>
      {datasets.map((ds, i) => (
        <DatasetRow key={ds.id} dataset={ds} onDelete={handleDelete} index={i} />
      ))}
    </tbody>
  </table>
)}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd ui && npx vitest run src/test/pages/DatasetsPage.test.tsx
```

Expected: 4 tests PASS

- [ ] **Step 7: Run full suite**

```bash
cd ui && npm test -- --run
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add ui/src/pages/DatasetsPage.tsx ui/src/test/pages/DatasetsPage.test.tsx
git commit -m "feat: mobile card view for datasets page"
```

---

## Task 6: Pipeline stepper mobile layout

**Files:**
- Modify: `ui/src/components/PipelineStages.tsx`
- Modify: `ui/src/test/components/PipelineStages.test.tsx`

The stepper uses `flex` with `min-w-[5.5rem]` per stage. On screens <400px this overflows. Fix: show a 2×2 grid on mobile, no connector lines (they don't translate to grid layout well).

- [ ] **Step 1: Check existing PipelineStages test file**

```bash
cat ui/src/test/components/PipelineStages.test.tsx
```

- [ ] **Step 2: Write failing tests**

If the file exists, append after the last `describe` block. If it does not exist, create it with this full content:

```typescript
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PipelineStages } from '@/components/PipelineStages'

const FOUR_STAGES = [
  { name: 'Generate', status: 'completed' as const },
  { name: 'Train', status: 'active' as const },
  { name: 'Evaluate', status: 'pending' as const },
  { name: 'Export', status: 'pending' as const },
]

describe('PipelineStages — desktop', () => {
  it('renders all stage names', () => {
    render(<PipelineStages stages={FOUR_STAGES} />)
    expect(screen.getByText('Generate')).toBeInTheDocument()
    expect(screen.getByText('Train')).toBeInTheDocument()
    expect(screen.getByText('Evaluate')).toBeInTheDocument()
    expect(screen.getByText('Export')).toBeInTheDocument()
  })
})

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

describe('PipelineStages — mobile', () => {
  it('renders all stage names on mobile', () => {
    mockMobile()
    render(<PipelineStages stages={FOUR_STAGES} />)
    expect(screen.getByText('Generate')).toBeInTheDocument()
    expect(screen.getByText('Train')).toBeInTheDocument()
    expect(screen.getByText('Evaluate')).toBeInTheDocument()
    expect(screen.getByText('Export')).toBeInTheDocument()
  })

  it('renders mobile grid container on mobile', () => {
    mockMobile()
    render(<PipelineStages stages={FOUR_STAGES} />)
    expect(screen.getByTestId('pipeline-mobile-grid')).toBeInTheDocument()
  })
})
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ui && npx vitest run src/test/components/PipelineStages.test.tsx
```

Expected: FAIL — `pipeline-mobile-grid` not found

- [ ] **Step 4: Add `useMediaQuery` import to `PipelineStages.tsx`**

```typescript
import { useMediaQuery } from '@/hooks/useMediaQuery'
```

- [ ] **Step 5: Replace `PipelineStages` function body**

Replace the entire `PipelineStages` export in `ui/src/components/PipelineStages.tsx`:

```tsx
export function PipelineStages({ stages, numbers }: PipelineStagesProps) {
  const isMobile = useMediaQuery('(max-width: 767px)')

  if (isMobile) {
    return (
      <div
        data-testid="pipeline-mobile-grid"
        className="grid grid-cols-2 gap-x-4 gap-y-5"
      >
        {stages.map((stage, i) => {
          const num = numbers?.[i] ?? String(i + 1).padStart(2, '0')
          return (
            <div
              key={stage.name}
              data-testid={`stage-${stage.name.toLowerCase().replace(/\s+/g, '-')}`}
              className={cn(
                'flex items-center gap-3',
                stage.status === 'pending' && 'opacity-40',
              )}
            >
              <StageNumber status={stage.status} label={num} />
              <span
                className={cn(
                  "font-['Outfit'] text-[0.72rem] uppercase tracking-[0.12em] font-medium",
                  stage.status === 'active' && 'text-[#1a1a1a]',
                  stage.status === 'completed' && 'text-[#888888]',
                  stage.status === 'pending' && 'text-[#b3b1a6]',
                  stage.status === 'failed' && 'text-[#7f1d1d]',
                )}
              >
                {stage.name}
              </span>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <div className="flex items-center w-full">
      {stages.map((stage, i) => {
        const num = numbers?.[i] ?? String(i + 1).padStart(2, '0')
        const isLast = i === stages.length - 1
        const prevDone =
          stage.status === 'completed' ||
          stages[i + 1]?.status === 'completed' ||
          stages[i + 1]?.status === 'active'
        return (
          <div
            key={stage.name}
            data-testid={`stage-${stage.name.toLowerCase().replace(/\s+/g, '-')}`}
            className={cn(
              'flex items-center',
              isLast ? 'flex-none' : 'flex-1',
              stage.status === 'pending' && 'opacity-40',
            )}
          >
            <div className="flex flex-col items-center gap-2 min-w-[5.5rem]">
              <StageNumber status={stage.status} label={num} />
              <span
                className={cn(
                  "font-['Outfit'] text-[0.72rem] uppercase tracking-[0.12em] font-medium",
                  stage.status === 'active' && 'text-[#1a1a1a]',
                  stage.status === 'completed' && 'text-[#888888]',
                  stage.status === 'pending' && 'text-[#b3b1a6]',
                  stage.status === 'failed' && 'text-[#7f1d1d]',
                )}
              >
                {stage.name}
              </span>
            </div>
            {!isLast && (
              <div
                aria-hidden
                className={cn(
                  'flex-1 h-px mx-2 transition-colors',
                  prevDone ? 'bg-[#1a1a1a]' : 'bg-[#d0d0c8]',
                )}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd ui && npx vitest run src/test/components/PipelineStages.test.tsx
```

Expected: all tests PASS

- [ ] **Step 7: Run full suite**

```bash
cd ui && npm test -- --run
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add ui/src/components/PipelineStages.tsx ui/src/test/components/PipelineStages.test.tsx
git commit -m "feat: responsive pipeline stepper with 2x2 grid on mobile"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|-------------|------|
| Mobile-friendly nav with hamburger | Task 2 |
| Responsive page padding | Task 3 |
| Models table → mobile cards | Task 4 |
| Datasets table → mobile cards | Task 5 |
| Pipeline stepper wraps on mobile | Task 6 |
| Tests validate all changes | All tasks |

### Gap check

- **RunsListPage**: Uses a list layout (not a table) — already works on mobile. No changes needed.
- **InferencePage / ModelDetailPage**: Use `grid grid-cols-1 lg:grid-cols-2` — already stacks on mobile.
- **ModelDetailPage PipelineHeader** (`ModelDetailPage.tsx:31`): Uses `PipelineStages` internally — covered by Task 6.
- **Forms** (ModelFormPage, DatasetsPage upload): Use `grid grid-cols-1 md:grid-cols-2` — already responsive.
- **Action button rows** (ModelDetailPage header, RunDetailPage header): Use `flex-wrap` — will wrap naturally on small screens.

### Placeholder scan

None. Every step contains runnable commands and complete code.

### Type consistency

- `useMediaQuery(query: string): boolean` — identical signature used in Tasks 4, 5, 6.
- `ModelMobileCard` props (`model`, `index`, `onRun`, `onDelete`, `deletePending`) — match call sites in Task 4 Step 5.
- `DatasetMobileCard` props (`dataset`, `onDelete`, `index`) — match `DatasetRow` signature and call sites in Task 5 Step 5.
- `data-testid="model-mobile-card"` — matches assertion `screen.getByTestId('model-mobile-card')` in Task 4 tests.
- `data-testid="dataset-mobile-card"` — matches assertion in Task 5 tests.
- `data-testid="pipeline-mobile-grid"` — matches assertion in Task 6 tests.
