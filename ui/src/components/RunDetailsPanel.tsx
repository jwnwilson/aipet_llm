import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'
import { getRunTemporal, getRunLogs, isRunActive } from '@/api/runs'
import { useLogStream } from '@/hooks/useLogStream'
import type { RunRecord, TemporalDetails, RunLogsResponse } from '@/types'

interface RunDetailsPanelProps {
  runId: string
  run: RunRecord
  datasetById?: Record<string, string>
}

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function TruncId({ id }: { id: string }) {
  const display = id.length > 22 ? `${id.slice(0, 9)}…${id.slice(-5)}` : id
  return <span title={id} className="block truncate">{display}</span>
}

function ProgressBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="flex items-center gap-2 mt-0.5">
      <div className="w-12 h-[3px] bg-[#f0efe8] rounded-full overflow-hidden shrink-0">
        <div className="h-full bg-[#1a1a1a] rounded-full" style={{ width: `${pct}%` }} />
      </div>
      <span>{pct}%</span>
    </div>
  )
}

function MetricCell({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-white px-3 py-2 min-w-0">
      <dt className="font-['IBM_Plex_Mono'] text-[0.57rem] uppercase tracking-[0.14em] text-[#888888] mb-0.5 truncate">
        {label}
      </dt>
      <dd className="font-['IBM_Plex_Mono'] text-[0.79rem] text-[#1a1a1a] break-all leading-snug">
        {value}
      </dd>
    </div>
  )
}

const GRID_COLS: Record<number, string> = {
  1: 'grid-cols-1',
  2: 'grid-cols-2',
  3: 'grid-cols-3',
  4: 'grid-cols-4',
}

function GridGroup({ children, cols = 4 }: { children: React.ReactNode; cols?: 1 | 2 | 3 | 4 }) {
  return (
    <div
      className={cn(
        'grid gap-px bg-[#e5e3d8] border border-[#e5e3d8] rounded-[3px] overflow-hidden mb-1.5',
        GRID_COLS[cols],
      )}
    >
      {children}
    </div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="font-['IBM_Plex_Mono'] text-[0.57rem] uppercase tracking-[0.18em] text-[#b3b1a6] mt-3 mb-1.5">
      {children}
    </div>
  )
}

function MetricsSection({ run, datasetById }: { run: RunRecord; datasetById: Record<string, string> }) {
  const hasDatasets = run.train_dataset_id != null || run.eval_dataset_id != null

  return (
    <div>
      <GridGroup cols={4}>
        <MetricCell label="Created" value={fmt(run.created_at)} />
        <MetricCell label="Updated" value={fmt(run.updated_at)} />
        <MetricCell
          label="Progress"
          value={run.progress != null
            ? <ProgressBar value={run.progress} />
            : <span className="text-[#b3b1a6]">—</span>
          }
        />
        <MetricCell
          label="Eval valid"
          value={run.eval_valid_pct != null
            ? `${Math.round(run.eval_valid_pct * 100)}%`
            : <span className="text-[#b3b1a6]">—</span>
          }
        />
      </GridGroup>

      {hasDatasets && (
        <GridGroup cols={2}>
          <MetricCell
            label="Train dataset"
            value={run.train_dataset_id != null
              ? (datasetById[run.train_dataset_id] ?? run.train_dataset_id)
              : <span className="text-[#b3b1a6]">—</span>
            }
          />
          <MetricCell
            label="Eval dataset"
            value={run.eval_dataset_id != null
              ? (datasetById[run.eval_dataset_id] ?? run.eval_dataset_id)
              : <span className="text-[#b3b1a6]">—</span>
            }
          />
        </GridGroup>
      )}

      {run.progress_detail && (
        <GridGroup cols={1}>
          <MetricCell label="Detail" value={run.progress_detail} />
        </GridGroup>
      )}
    </div>
  )
}

function TemporalSection({ details }: { details: TemporalDetails }) {
  return (
    <div>
      <SectionLabel>Workflow</SectionLabel>
      <GridGroup cols={2}>
        <MetricCell label="Workflow ID" value={<TruncId id={details.workflow_id} />} />
        <MetricCell label="Temporal Run ID" value={<TruncId id={details.temporal_run_id} />} />
        <MetricCell label="Status" value={details.status} />
        {details.start_time != null && (
          <MetricCell label="Started" value={fmt(details.start_time)} />
        )}
        {details.close_time != null && (
          <MetricCell label="Finished" value={fmt(details.close_time)} />
        )}
      </GridGroup>
    </div>
  )
}

function LogsSection({ logsData }: { logsData: RunLogsResponse }) {
  return (
    <div>
      <SectionLabel>Training logs</SectionLabel>
      {logsData.logs != null ? (
        <pre className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#1a1a1a] bg-[#f6f5f0] border border-[#d0d0c8] rounded-[2px] p-3 overflow-x-auto whitespace-pre-wrap break-all max-h-56">
          {logsData.logs}
        </pre>
      ) : (
        <p className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#888888] italic">
          No logs captured
        </p>
      )}
    </div>
  )
}

export function RunDetailsPanel({ runId, run, datasetById = {} }: RunDetailsPanelProps) {
  const [expanded, setExpanded] = useState(true)
  const isActive = isRunActive(run)
  const pollInterval = isActive ? 5000 : false

  const { data: temporalData, isError: temporalError } = useQuery({
    queryKey: ['runs', runId, 'temporal'],
    queryFn: () => getRunTemporal(runId),
    enabled: expanded,
    refetchInterval: expanded ? pollInterval : false,
  })

  const { data: logsData } = useQuery({
    queryKey: ['runs', runId, 'logs'],
    queryFn: () => getRunLogs(runId),
    enabled: expanded && !isActive,
    refetchInterval: false,
  })

  const { lines: streamedLines } = useLogStream(runId, isActive && expanded)

  return (
    <div className="border-t border-[#e5e3d8] mt-3 pt-1.5">
      <button
        type="button"
        onClick={() => setExpanded(prev => !prev)}
        className="flex items-center gap-1.5 font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em] text-[#888888] hover:text-[#1a1a1a] py-0.5"
        aria-expanded={expanded}
      >
        {expanded
          ? <ChevronUp className="h-3 w-3" aria-hidden />
          : <ChevronDown className="h-3 w-3" aria-hidden />
        }
        Details
      </button>

      {expanded && (
        <div className="mt-2">
          <MetricsSection run={run} datasetById={datasetById} />

          {run.training_config != null && Object.keys(run.training_config).length > 0 && (
            <div>
              <SectionLabel>Config</SectionLabel>
              <GridGroup cols={3}>
                {Object.entries(run.training_config)
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => (
                    <MetricCell key={k} label={k.replace(/_/g, ' ')} value={String(v)} />
                  ))}
              </GridGroup>
            </div>
          )}

          {temporalError && (
            <p className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#7f1d1d] mb-2">
              Failed to load workflow details
            </p>
          )}
          {temporalData != null && <TemporalSection details={temporalData} />}

          {!isActive && logsData != null && <LogsSection logsData={logsData} />}

          {isActive && (
            <div>
              <SectionLabel>Training logs — live</SectionLabel>
              <pre className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#1a1a1a] bg-[#f6f5f0] border border-[#d0d0c8] rounded-[2px] p-3 overflow-x-auto whitespace-pre-wrap break-all max-h-56">
                {streamedLines.length > 0 ? streamedLines.join('\n') : (
                  <span className="text-[#888888] italic font-['DM_Serif_Display']">Waiting for output…</span>
                )}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
