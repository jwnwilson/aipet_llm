import { useEffect, useState } from 'react'
import { getAuthToken } from '@/api/client'

interface UseLogStreamResult {
  lines: string[]
}

export function useLogStream(runId: string, active: boolean): UseLogStreamResult {
  const [lines, setLines] = useState<string[]>([])

  useEffect(() => {
    if (!active) return

    setLines([])
    const controller = new AbortController()

    async function stream() {
      const token = await getAuthToken()
      const headers: Record<string, string> = {}
      if (token) headers['Authorization'] = `Bearer ${token}`

      const baseUrl = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
      let res: Response
      try {
        res = await fetch(`${baseUrl}/api/runs/${runId}/logs/stream`, {
          headers,
          signal: controller.signal,
        })
      } catch {
        return
      }
      if (!res.ok || !res.body) return

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const parts = buffer.split('\n\n')
          buffer = parts.pop() ?? ''
          for (const part of parts) {
            if (part.includes('event: done')) {
              return
            }
            const match = part.match(/^data: (.*)$/m)
            if (match) {
              setLines(prev => [...prev, match[1]])
            }
          }
        }
      } finally {
        reader.cancel()
      }
    }

    stream().catch(() => {})

    return () => {
      controller.abort()
    }
  }, [runId, active])

  return { lines }
}
