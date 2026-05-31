import { renderHook, act } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { useLogStream } from '@/hooks/useLogStream'

class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  onmessage: ((e: { data: string }) => void) | null = null
  addEventListener = vi.fn((event: string, cb: () => void) => {
    if (event === 'done') this._doneHandler = cb
  })
  close = vi.fn()
  _doneHandler: (() => void) | null = null

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  emit(data: string) {
    this.onmessage?.({ data })
  }
}

beforeEach(() => {
  MockEventSource.instances = []
  vi.stubGlobal('EventSource', MockEventSource)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useLogStream', () => {
  it('returns empty lines when active=false', () => {
    const { result } = renderHook(() => useLogStream('run-1', false))
    expect(result.current.lines).toEqual([])
  })

  it('does not open EventSource when active=false', () => {
    renderHook(() => useLogStream('run-1', false))
    expect(MockEventSource.instances).toHaveLength(0)
  })

  it('opens EventSource when active=true', () => {
    renderHook(() => useLogStream('run-1', true))
    expect(MockEventSource.instances).toHaveLength(1)
    expect(MockEventSource.instances[0].url).toContain('/api/runs/run-1/logs/stream')
  })

  it('accumulates lines from onmessage events', () => {
    const { result } = renderHook(() => useLogStream('run-1', true))
    act(() => { MockEventSource.instances[0].emit('epoch 1') })
    act(() => { MockEventSource.instances[0].emit('epoch 2') })
    expect(result.current.lines).toEqual(['epoch 1', 'epoch 2'])
  })

  it('closes EventSource on unmount', () => {
    const { unmount } = renderHook(() => useLogStream('run-1', true))
    unmount()
    expect(MockEventSource.instances[0].close).toHaveBeenCalled()
  })
})
