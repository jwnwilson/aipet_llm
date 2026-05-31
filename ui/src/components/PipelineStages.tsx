import { Check, X, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useMediaQuery } from '@/hooks/useMediaQuery'

export type StageStatus = 'pending' | 'active' | 'completed' | 'failed'

export interface PipelineStage {
  name: string
  status: StageStatus
}

interface PipelineStagesProps {
  stages: PipelineStage[]
  /** Optional explicit numbering — if omitted, uses 1-based index */
  numbers?: string[]
}

function StageNumber({ status, label }: { status: StageStatus; label: string }) {
  const base = 'inline-flex items-center justify-center h-9 w-9 rounded-full text-[0.78rem] font-medium select-none transition-colors'
  if (status === 'active') {
    return (
      <span className={cn(base, "bg-[#1a1a1a] text-[#fafaf7] border-[1.5px] border-[#1a1a1a] font-['IBM_Plex_Mono']")}>
        <Loader2 className="h-4 w-4 animate-spin" aria-label="active" />
      </span>
    )
  }
  if (status === 'completed') {
    return (
      <span className={cn(base, 'bg-[#f3f2ec] text-[#888888] border-[1.5px] border-[#b3b1a6]')}>
        <Check className="h-4 w-4" aria-label="completed" />
      </span>
    )
  }
  if (status === 'failed') {
    return (
      <span className={cn(base, 'bg-[#7f1d1d] text-[#fafaf7] border-[1.5px] border-[#7f1d1d]')}>
        <X className="h-4 w-4" aria-label="failed" />
      </span>
    )
  }
  // pending — show outlined number
  return (
    <span className={cn(base, "bg-white text-[#b3b1a6] border-[1.5px] border-[#d0d0c8] font-['IBM_Plex_Mono']")}>
      {label}
    </span>
  )
}

export function PipelineStages({ stages, numbers }: PipelineStagesProps) {
  const isMobile = useMediaQuery('(max-width: 767px)')

  if (isMobile) {
    return (
      <div
        data-testid="pipeline-mobile-grid"
        className="grid grid-cols-2 gap-x-4 gap-y-5"
      >
        {stages.map((stage, i) => {
          const num = numbers?.[i] ?? String(i + 1).padStart(2, '0')
          return (
            <div
              key={stage.name}
              data-testid={`stage-${stage.name.toLowerCase().replace(/\s+/g, '-')}`}
              className={cn(
                'flex items-center gap-3',
                stage.status === 'pending' && 'opacity-40',
              )}
            >
              <StageNumber status={stage.status} label={num} />
              <span
                className={cn(
                  "font-['Outfit'] text-[0.72rem] uppercase tracking-[0.12em] font-medium",
                  stage.status === 'active' && 'text-[#1a1a1a]',
                  stage.status === 'completed' && 'text-[#888888]',
                  stage.status === 'pending' && 'text-[#b3b1a6]',
                  stage.status === 'failed' && 'text-[#7f1d1d]',
                )}
              >
                {stage.name}
              </span>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <div className="flex items-center w-full">
      {stages.map((stage, i) => {
        const num = numbers?.[i] ?? String(i + 1).padStart(2, '0')
        const isLast = i === stages.length - 1
        const prevDone = stage.status === 'completed' || stages[i + 1]?.status === 'completed' || stages[i + 1]?.status === 'active'
        return (
          <div
            key={stage.name}
            data-testid={`stage-${stage.name.toLowerCase().replace(/\s+/g, '-')}`}
            className={cn(
              'flex items-center',
              isLast ? 'flex-none' : 'flex-1',
              stage.status === 'pending' && 'opacity-40',
            )}
          >
            <div className="flex flex-col items-center gap-2 min-w-[5.5rem]">
              <StageNumber status={stage.status} label={num} />
              <span
                className={cn(
                  "font-['Outfit'] text-[0.72rem] uppercase tracking-[0.12em] font-medium",
                  stage.status === 'active' && 'text-[#1a1a1a]',
                  stage.status === 'completed' && 'text-[#888888]',
                  stage.status === 'pending' && 'text-[#b3b1a6]',
                  stage.status === 'failed' && 'text-[#7f1d1d]',
                )}
              >
                {stage.name}
              </span>
            </div>
            {!isLast && (
              <div
                aria-hidden
                className={cn(
                  'flex-1 h-px mx-2 transition-colors',
                  prevDone ? 'bg-[#1a1a1a]' : 'bg-[#d0d0c8]',
                )}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

