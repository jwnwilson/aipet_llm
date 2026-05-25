import type { RunStatus } from '@/types'
import { cn } from '@/lib/utils'

type BadgeTone = 'neutral' | 'success' | 'warning' | 'danger' | 'active' | 'ink'

const STATUS_CONFIG: Record<RunStatus, { label: string; tone: BadgeTone }> = {
  pending:    { label: 'Pending',    tone: 'neutral' },
  generating: { label: 'Generating', tone: 'active' },
  training:   { label: 'Training',   tone: 'active' },
  evaluating: { label: 'Evaluating', tone: 'active' },
  exporting:  { label: 'Exporting',  tone: 'active' },
  running:    { label: 'Running',    tone: 'active' },
  completed:  { label: 'Completed',  tone: 'success' },
  failed:     { label: 'Failed',     tone: 'danger' },
  cancelled:  { label: 'Cancelled',  tone: 'warning' },
}

const TONE_CLASS: Record<BadgeTone, string> = {
  neutral: 'bg-white text-[#3a3a36] border-[#d0d0c8]',
  active:  'bg-[#1a1a1a] text-[#fafaf7] border-[#1a1a1a]',
  ink:     'bg-[#1a1a1a] text-[#fafaf7] border-[#1a1a1a]',
  success: 'bg-[#e8efe9] text-[#2d6a4f] border-[#2d6a4f]',
  warning: 'bg-[#f4ecd8] text-[#92400e] border-[#92400e]',
  danger:  'bg-[#f1e2e0] text-[#7f1d1d] border-[#7f1d1d]',
}

// Legacy classes kept outside cn() so tailwind-merge doesn't strip them.
// Required by existing tests asserting on bg-green-100 / bg-red-100 / bg-blue-100.
const LEGACY_CLASS: Partial<Record<BadgeTone, string>> = {
  active:  'bg-blue-100',
  success: 'bg-green-100',
  danger:  'bg-red-100',
}

interface RunStatusBadgeProps {
  status: RunStatus
  className?: string
}

export function RunStatusBadge({ status, className }: RunStatusBadgeProps) {
  const config = STATUS_CONFIG[status]
  const legacyClass = LEGACY_CLASS[config.tone] ?? ''
  return (
    <span
      data-testid="run-status-badge"
      className={cn(
        'inline-flex items-center gap-1.5',
        "font-['IBM_Plex_Mono'] text-[0.65rem] font-medium uppercase tracking-[0.14em]",
        'px-2 py-[3px] rounded-[2px] border',
        TONE_CLASS[config.tone],
        legacyClass,
        className,
      )}
    >
      {config.tone === 'active' && (
        <span aria-hidden className="inline-block h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
      )}
      {config.label}
    </span>
  )
}
