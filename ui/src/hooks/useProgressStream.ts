import { useEffect, useState } from 'react'
import { getAuthToken } from '@/api/client'

interface ProgressData {
  fraction: number
  detail: string
}

interface UseProgressStreamResult {
  progress: ProgressData | null
}

export function useProgressStream(runId: string, active: boolean): UseProgressStreamResult {
  const [progress, setProgress] = useState<ProgressData | null>(null)

  useEffect(() => {
    if (!active) {
      setProgress(null)
      return
    }

    const controller = new AbortController()

    async function stream() {
      const token = await getAuthToken()
      const headers: Record<string, string> = {}
      if (token) headers['Authorization'] = `Bearer ${token}`

      const baseUrl = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
      let res: Response
      try {
        res = await fetch(`${baseUrl}/api/runs/${runId}/progress/stream`, {
          headers,
          cache: 'no-store',
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
            if (part.includes('event: done')) return
            const match = part.match(/^data: (.*)$/m)
            if (match) {
              try {
                setProgress(JSON.parse(match[1]) as ProgressData)
              } catch {
                // skip malformed events
              }
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

  return { progress }
}
