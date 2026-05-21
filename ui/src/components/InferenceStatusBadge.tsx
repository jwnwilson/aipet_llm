import type { InferenceStatus } from '../types'
import { cn } from '../lib/utils'

const STATUS_CONFIG: Record<InferenceStatus, { label: string; className: string }> = {
  pending:      { label: 'Pending',      className: 'bg-gray-100 text-gray-600' },
  initializing: { label: 'Initializing', className: 'bg-amber-100 text-amber-800' },
  available:    { label: 'Available',    className: 'bg-green-100 text-green-800' },
  idle:         { label: 'Idle',         className: 'bg-blue-100 text-blue-800' },
  shutdown:     { label: 'Shutdown',     className: 'bg-slate-100 text-slate-600' },
  failed:       { label: 'Failed',       className: 'bg-red-100 text-red-800' },
}

interface InferenceStatusBadgeProps {
  status: InferenceStatus
  className?: string
}

export function InferenceStatusBadge({ status, className }: InferenceStatusBadgeProps) {
  const config = STATUS_CONFIG[status]
  return (
    <span
      data-testid="inference-status-badge"
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium',
        config.className,
        className,
      )}
    >
      {config.label}
    </span>
  )
}
