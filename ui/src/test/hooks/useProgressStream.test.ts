import { renderHook, waitFor } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { useProgressStream } from '@/hooks/useProgressStream'

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

describe('useProgressStream', () => {
  it('returns null progress when active=false', () => {
    const { result } = renderHook(() => useProgressStream('run-1', false))
    expect(result.current.progress).toBeNull()
  })

  it('does not call fetch when active=false', () => {
    renderHook(() => useProgressStream('run-1', false))
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('calls fetch for the progress/stream endpoint when active=true', async () => {
    renderHook(() => useProgressStream('run-1', true))
    await waitFor(() => expect(mockFetch).toHaveBeenCalledOnce())
    expect(mockFetch.mock.calls[0][0]).toContain('/api/runs/run-1/progress/stream')
  })

  it('parses a progress event and updates state', async () => {
    mockFetch.mockResolvedValue(
      makeSseResponse('data: {"fraction": 0.42, "detail": "step 42/100"}\n\nevent: done\ndata: stream closed\n\n')
    )
    const { result } = renderHook(() => useProgressStream('run-1', true))
    await waitFor(() =>
      expect(result.current.progress).toEqual({ fraction: 0.42, detail: 'step 42/100' })
    )
  })

  it('updates progress across multiple events, keeping latest', async () => {
    mockFetch.mockResolvedValue(
      makeSseResponse(
        'data: {"fraction": 0.3, "detail": "step 30/100"}\n\n' +
        'data: {"fraction": 0.6, "detail": "step 60/100"}\n\n' +
        'event: done\ndata: stream closed\n\n'
      )
    )
    const { result } = renderHook(() => useProgressStream('run-1', true))
    await waitFor(() =>
      expect(result.current.progress).toEqual({ fraction: 0.6, detail: 'step 60/100' })
    )
  })

  it('resets progress to null when active switches to false', async () => {
    mockFetch.mockResolvedValue(
      makeSseResponse('data: {"fraction": 0.7, "detail": "nearly done"}\n\n')
    )
    const { result, rerender } = renderHook(
      ({ active }: { active: boolean }) => useProgressStream('run-2', active),
      { initialProps: { active: true } }
    )
    await waitFor(() => expect(result.current.progress).not.toBeNull())
    rerender({ active: false })
    expect(result.current.progress).toBeNull()
  })

  it('ignores malformed JSON events and continues to parse valid ones', async () => {
    mockFetch.mockResolvedValue(
      makeSseResponse(
        'data: not-valid-json\n\n' +
        'data: {"fraction": 0.9, "detail": "recovered"}\n\n' +
        'event: done\ndata: stream closed\n\n'
      )
    )
    const { result } = renderHook(() => useProgressStream('run-3', true))
    await waitFor(() => expect(result.current.progress?.fraction).toBe(0.9))
    expect(result.current.progress?.detail).toBe('recovered')
  })

  it('aborts the request on unmount', async () => {
    const abortSpy = vi.spyOn(AbortController.prototype, 'abort')
    const { unmount } = renderHook(() => useProgressStream('run-1', true))
    await waitFor(() => expect(mockFetch).toHaveBeenCalled())
    unmount()
    expect(abortSpy).toHaveBeenCalled()
    abortSpy.mockRestore()
  })
})
