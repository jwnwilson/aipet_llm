import { renderHook, waitFor } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { useLogStream } from '@/hooks/useLogStream'

vi.mock('@/api/client', () => ({
  getAuthToken: vi.fn().mockResolvedValue(null),
}))

function makeSseResponse(sseText: string): { ok: boolean; body: ReadableStream<Uint8Array> } {
  const encoder = new TextEncoder()
  return {
    ok: true,
    body: new ReadableStream({
      start(controller) {
        if (sseText) controller.enqueue(encoder.encode(sseText))
        controller.close()
      },
    }),
  }
}

let mockFetch: ReturnType<typeof vi.fn>

beforeEach(() => {
  mockFetch = vi.fn().mockResolvedValue(makeSseResponse(''))
  vi.stubGlobal('fetch', mockFetch)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

describe('useLogStream', () => {
  it('returns empty lines when active=false', () => {
    const { result } = renderHook(() => useLogStream('run-1', false))
    expect(result.current.lines).toEqual([])
  })

  it('does not call fetch when active=false', () => {
    renderHook(() => useLogStream('run-1', false))
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('calls fetch for the logs/stream endpoint when active=true', async () => {
    renderHook(() => useLogStream('run-1', true))
    await waitFor(() => expect(mockFetch).toHaveBeenCalledOnce())
    expect(mockFetch.mock.calls[0][0]).toContain('/api/runs/run-1/logs/stream')
  })

  it('accumulates lines from SSE data events', async () => {
    mockFetch.mockResolvedValue(
      makeSseResponse('data: epoch 1\n\ndata: epoch 2\n\nevent: done\ndata: end\n\n')
    )
    const { result } = renderHook(() => useLogStream('run-1', true))
    await waitFor(() => expect(result.current.lines).toEqual(['epoch 1', 'epoch 2']))
  })

  it('aborts the request on unmount', async () => {
    const abortSpy = vi.spyOn(AbortController.prototype, 'abort')
    const { unmount } = renderHook(() => useLogStream('run-1', true))
    await waitFor(() => expect(mockFetch).toHaveBeenCalled())
    unmount()
    expect(abortSpy).toHaveBeenCalled()
    abortSpy.mockRestore()
  })
})
