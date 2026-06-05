import { useEffect, useRef } from 'react'

interface RunLogViewerProps {
  logs: string
  isLive?: boolean
}

/**
 * Editorial log viewer — paper-white background, ink text, fixed-width.
 * No terminal-green theming; this is a scientific journal aesthetic.
 */
export function RunLogViewer({ logs, isLive = false }: RunLogViewerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <div className="bg-white border border-[#d0d0c8] rounded-[4px]">
      <div className="px-4 py-2 border-b border-[#d0d0c8] flex items-center justify-between bg-[#f6f5ef]">
        <span className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
          stdout · log stream
        </span>
        {isLive && (
          <span className="inline-flex items-center gap-1.5 font-['IBM_Plex_Mono'] text-[0.65rem] text-[#6b6b6b]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#2d6a4f] animate-pulse" />
            Live
          </span>
        )}
      </div>
      <div className="relative h-64 overflow-y-auto p-4">
        <pre className="font-['IBM_Plex_Mono'] text-[0.78rem] leading-[1.55] text-[#1a1a1a] whitespace-pre-wrap break-words">
          {logs || (
            <span className="text-[#767676] italic font-['DM_Serif_Display']">No output yet.</span>
          )}
        </pre>
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
