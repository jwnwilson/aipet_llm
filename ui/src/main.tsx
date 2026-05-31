import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Auth0Provider } from '@auth0/auth0-react'
import { BrowserRouter } from 'react-router-dom'
import { MockAuth0Provider } from './auth/MockAuth0Provider'
import './index.css'
import App from './App.tsx'

const authDisabled = import.meta.env.VITE_AUTH_DISABLED === 'true'

if (!authDisabled) {
  const _domain = import.meta.env.VITE_AUTH0_DOMAIN
  const _clientId = import.meta.env.VITE_AUTH0_CLIENT_ID
  if (!_domain || !_clientId) {
    throw new Error('Missing VITE_AUTH0_DOMAIN or VITE_AUTH0_CLIENT_ID — copy .env.local.example to .env.local')
  }
}

const Provider = authDisabled
  ? ({ children }: { children: React.ReactNode }) => <MockAuth0Provider>{children}</MockAuth0Provider>
  : ({ children }: { children: React.ReactNode }) => (
      <Auth0Provider
        domain={import.meta.env.VITE_AUTH0_DOMAIN}
        clientId={import.meta.env.VITE_AUTH0_CLIENT_ID}
        authorizationParams={{
          redirect_uri: window.location.origin,
          audience: import.meta.env.VITE_AUTH0_AUDIENCE,
        }}
      >
        {children}
      </Auth0Provider>
    )

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Provider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </Provider>
  </StrictMode>,
)
