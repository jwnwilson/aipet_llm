import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { getRunTemporal, getRunLogs, isRunActive } from '@/api/runs'
import { useLogStream } from '@/hooks/useLogStream'
import type { RunRecord, TemporalDetails, RunLogsResponse } from '@/types'

interface RunDetailsPanelProps {
  runId: string
  run: RunRecord
  datasetById?: Record<string, string>
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 py-2 border-b border-[#e5e3d8] last:border-0">
      <dt className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.14em] text-[#888888]">
        {label}
      </dt>
      <dd className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a] break-all">
        {value}
      </dd>
    </div>
  )
}

function MetricsSection({ run, datasetById }: { run: RunRecord; datasetById: Record<string, string> }) {
  return (
    <div className="mb-4">
      <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.18em] text-[#888888] mb-2">
        Metrics
      </div>
      <dl>
        <DetailRow label="Run ID" value={run.id} />
        <DetailRow label="Created" value={new Date(run.created_at).toLocaleString()} />
        <DetailRow label="Updated" value={new Date(run.updated_at).toLocaleString()} />
        {run.progress != null && (
          <DetailRow label="Progress" value={`${Math.round(run.progress * 100)}%`} />
        )}
        {run.eval_valid_pct != null && (
          <DetailRow label="Eval valid" value={`${Math.round(run.eval_valid_pct * 100)}%`} />
        )}
        {run.progress_detail && (
          <DetailRow label="Detail" value={run.progress_detail} />
        )}
        {run.train_dataset_id != null && (
          <DetailRow
            label="Train dataset"
            value={datasetById[run.train_dataset_id] ?? run.train_dataset_id}
          />
        )}
        {run.eval_dataset_id != null && (
          <DetailRow
            label="Eval dataset"
            value={datasetById[run.eval_dataset_id] ?? run.eval_dataset_id}
          />
        )}
      </dl>
    </div>
  )
}

function TemporalSection({ details }: { details: TemporalDetails }) {
  return (
    <div className="mb-4">
      <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.18em] text-[#888888] mb-2">
        Workflow
      </div>
      <dl>
        <DetailRow label="Workflow ID" value={details.workflow_id} />
        <DetailRow label="Temporal Run ID" value={details.temporal_run_id} />
        <DetailRow label="Status" value={details.status} />
        {details.start_time != null && (
          <DetailRow
            label="Started"
            value={new Date(details.start_time).toLocaleString()}
          />
        )}
        {details.close_time != null && (
          <DetailRow
            label="Finished"
            value={new Date(details.close_time).toLocaleString()}
          />
        )}
      </dl>
    </div>
  )
}

function LogsSection({ logsData }: { logsData: RunLogsResponse }) {
  return (
    <div>
      <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.18em] text-[#888888] mb-2">
        Training logs
      </div>
      {logsData.logs != null ? (
        <pre className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#1a1a1a] bg-[#f6f5f0] border border-[#d0d0c8] rounded-[2px] p-3 overflow-x-auto whitespace-pre-wrap break-all max-h-64">
          {logsData.logs}
        </pre>
      ) : (
        <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#888888] italic">
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
    <div className="border-t border-[#e5e3d8] mt-4 pt-2">
      <button
        type="button"
        onClick={() => setExpanded(prev => !prev)}
        className="flex items-center gap-1.5 font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] hover:text-[#1a1a1a] py-1"
        aria-expanded={expanded}
      >
        {expanded
          ? <ChevronUp className="h-3 w-3" aria-hidden />
          : <ChevronDown className="h-3 w-3" aria-hidden />
        }
        Stage details
      </button>

      {expanded && (
        <div className="mt-3 pb-1">
          <MetricsSection run={run} datasetById={datasetById} />

          {run.training_config != null && Object.keys(run.training_config).length > 0 && (
            <div className="mb-4">
              <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.18em] text-[#888888] mb-2">
                Run configuration
              </div>
              <dl>
                {Object.entries(run.training_config)
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => (
                    <DetailRow key={k} label={k.replace(/_/g, ' ')} value={String(v)} />
                  ))}
              </dl>
            </div>
          )}

          {temporalError && (
            <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d] mb-3">
              Failed to load workflow details
            </p>
          )}
          {temporalData != null && <TemporalSection details={temporalData} />}

          {/* Static logs for completed/failed runs */}
          {!isActive && logsData != null && <LogsSection logsData={logsData} />}

          {/* Live stream for active runs */}
          {isActive && (
            <div>
              <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.18em] text-[#888888] mb-2">
                Training logs — live
              </div>
              <pre className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#1a1a1a] bg-[#f6f5f0] border border-[#d0d0c8] rounded-[2px] p-3 overflow-x-auto whitespace-pre-wrap break-all max-h-64">
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
