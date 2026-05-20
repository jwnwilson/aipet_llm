/**
 * MockAuth0Provider — replaces Auth0Provider when VITE_AUTH_DISABLED=true.
 *
 * Presents a fake authenticated admin user so the rest of the app works
 * without an Auth0 tenant.  Never use this in production.
 */
import type { ReactNode } from 'react'
import { Auth0Context } from '@auth0/auth0-react'

const ROLES_CLAIM = 'https://aipet/roles'

const LOCAL_USER = {
  email: 'local@dev',
  name: 'Local Dev',
  sub: 'local|dev',
  [ROLES_CLAIM]: ['admin'],
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockCtx: any = {
  isAuthenticated: true,
  isLoading: false,
  error: undefined,
  user: LOCAL_USER,
  loginWithRedirect: async () => {},
  loginWithPopup: async () => {},
  logout: async () => {},
  // Return empty string — apiClient skips the Authorization header for empty tokens
  getAccessTokenSilently: async () => '',
  getAccessTokenWithPopup: async () => undefined,
  getIdTokenClaims: async () => undefined,
  handleRedirectCallback: async () => ({ appState: undefined }),
}

export function MockAuth0Provider({ children }: { children: ReactNode }) {
  return (
    <Auth0Context.Provider value={mockCtx}>
      {children}
    </Auth0Context.Provider>
  )
}
