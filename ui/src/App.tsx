import { type ReactNode, useEffect, useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Link, Navigate, NavLink, Route, Routes } from 'react-router-dom'
import { ModelsListPage } from './pages/ModelsListPage'
import { ModelFormPage } from './pages/ModelFormPage'
import { ModelDetailPage } from './pages/ModelDetailPage'
import { RunsListPage } from './pages/RunsListPage'
import { RunDetailPage } from './pages/RunDetailPage'
import { DatasetsPage } from './pages/DatasetsPage'
import { InferencePage } from './pages/InferencePage'
import { UsersPage } from './pages/UsersPage'
import { TokenSync } from './components/TokenSync'
import { AccessPending } from './components/AccessPending'

const queryClient = new QueryClient()

const ROLES_CLAIM = 'https://aipet/roles'

function useIsAdmin(): boolean {
  const { user } = useAuth0()
  const roles: string[] = user?.[ROLES_CLAIM] ?? []
  return roles.includes('admin')
}

function AuthCluster() {
  const { logout, user } = useAuth0()
  return (
    <div className="flex items-center gap-4 ml-auto">
      <span className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#888888] tracking-[0.04em] hidden sm:inline">
        {user?.email}
      </span>
      <button
        onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
        className="font-['Outfit'] text-[0.72rem] font-medium uppercase tracking-[0.12em] text-[#888888] hover:text-[#1a1a1a] transition-colors"
      >
        Logout
      </button>
    </div>
  )
}

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

function AdminRoute({ children }: { children: ReactNode }) {
  const { isLoading } = useAuth0()
  const isAdmin = useIsAdmin()
  if (isLoading) return null
  return isAdmin ? <>{children}</> : <Navigate to="/models" replace />
}

function LoadingScreen() {
  return (
    <div className="flex items-center justify-center h-screen">
      <div className="flex flex-col items-center gap-3">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
          Loading
        </span>
        <div className="h-px w-24 bg-[#1a1a1a] animate-pulse" />
      </div>
    </div>
  )
}

function AppContent() {
  const { isAuthenticated, isLoading, loginWithRedirect, error } = useAuth0()
  const [accessDenied, setAccessDenied] = useState(false)

  useEffect(() => {
    const handler = () => setAccessDenied(true)
    window.addEventListener('auth:access-denied', handler)
    return () => window.removeEventListener('auth:access-denied', handler)
  }, [])

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      loginWithRedirect()
    }
  }, [isLoading, isAuthenticated, loginWithRedirect])

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="max-w-md text-center">
          <h2 className="font-['DM_Serif_Display'] text-2xl text-[#7f1d1d] mb-2">
            Authentication Error
          </h2>
          <p className="font-['IBM_Plex_Mono'] text-sm text-[#888888]">{error.message}</p>
        </div>
      </div>
    )
  }

  if (accessDenied) return <AccessPending />

  if (isLoading || !isAuthenticated) {
    return <LoadingScreen />
  }

  return (
    <div className="min-h-screen">
      <TokenSync />
      <Nav />
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/models" replace />} />
          <Route path="/models" element={<ModelsListPage />} />
          <Route path="/models/new" element={<ModelFormPage />} />
          <Route path="/models/:id" element={<ModelDetailPage />} />
          <Route path="/models/:id/edit" element={<ModelFormPage />} />
          <Route path="/runs" element={<RunsListPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/inferences" element={<InferencePage />} />
          <Route path="/admin/users" element={<AdminRoute><UsersPage /></AdminRoute>} />
        </Routes>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  )
}
