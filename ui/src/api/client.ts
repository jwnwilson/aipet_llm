import axios from 'axios'

let _tokenGetter: (() => Promise<string>) | null = null

export function setTokenGetter(fn: (() => Promise<string>) | null): void {
  _tokenGetter = fn
}

export async function getAuthToken(): Promise<string | null> {
  if (!_tokenGetter) return null
  try {
    return await _tokenGetter()
  } catch {
    return null
  }
}

// Do NOT set a default Content-Type here.
// Axios sets 'application/json' automatically for plain-object bodies.
// For FormData bodies (file uploads), axios auto-sets 'multipart/form-data'
// with the correct boundary — a hardcoded default would override that and
// cause FastAPI to reject uploads with a 422 (all form fields null).
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '',
})

apiClient.interceptors.request.use(async (config) => {
  if (_tokenGetter) {
    try {
      const token = await _tokenGetter()
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
    } catch (err) {
      console.error('[apiClient] Token getter failed — sending request without auth', err)
    }
  }
  return config
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 403) {
      window.dispatchEvent(new CustomEvent('auth:access-denied'))
    }
    return Promise.reject(error)
  }
)
