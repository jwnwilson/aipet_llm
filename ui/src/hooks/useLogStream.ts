import { useEffect, useState } from 'react'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

interface UseLogStreamResult {
  lines: string[]
}

export function useLogStream(runId: string, active: boolean): UseLogStreamResult {
  const [lines, setLines] = useState<string[]>([])

  useEffect(() => {
    if (!active) return

    setLines([])
    const es = new EventSource(`${API_BASE}/api/runs/${runId}/logs/stream`)

    es.onmessage = (e: MessageEvent<string>) => {
      setLines(prev => [...prev, e.data])
    }

    es.addEventListener('done', () => {
      es.close()
    })

    return () => {
      es.close()
    }
  }, [runId, active])

  return { lines }
}
