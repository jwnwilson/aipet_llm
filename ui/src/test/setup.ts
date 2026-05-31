import '@testing-library/jest-dom/vitest'
import { afterAll, afterEach, beforeAll, vi } from 'vitest'
import { server } from './msw/server'
import { resetHandlerState } from './msw/handlers'

window.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} }

// jsdom does not implement EventSource — provide a no-op stub so components
// that use useLogStream don't throw during tests that render active runs.
if (typeof window.EventSource === 'undefined') {
  class EventSourceStub {
    static readonly CONNECTING = 0
    static readonly OPEN = 1
    static readonly CLOSED = 2
    readonly CONNECTING = 0
    readonly OPEN = 1
    readonly CLOSED = 2
    url: string
    onmessage: ((e: MessageEvent) => void) | null = null
    onerror: ((e: Event) => void) | null = null
    onopen: ((e: Event) => void) | null = null
    constructor(url: string) { this.url = url }
    addEventListener() {}
    removeEventListener() {}
    dispatchEvent() { return false }
    close() {}
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).EventSource = EventSourceStub
}
Element.prototype.scrollIntoView ??= () => {}
Element.prototype.hasPointerCapture ??= () => false

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

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => {
  server.resetHandlers()
  resetHandlerState()
})
afterAll(() => server.close())
