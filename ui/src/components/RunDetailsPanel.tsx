import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { getRunTemporal, getRunLogs, isRunActive } from '@/api/runs'
import type { RunRecord, TemporalDetails, RunLogsResponse } from '@/types'

interface RunDetailsPanelProps {
  runId: string
  run: RunRecord
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

export function RunDetailsPanel({ runId, run }: RunDetailsPanelProps) {
  const [expanded, setExpanded] = useState(false)
  const pollInterval = isRunActive(run) ? 5000 : false

  const { data: temporalData, isError: temporalError } = useQuery({
    queryKey: ['runs', runId, 'temporal'],
    queryFn: () => getRunTemporal(runId),
    enabled: expanded,
    refetchInterval: expanded ? pollInterval : false,
  })

  const { data: logsData } = useQuery({
    queryKey: ['runs', runId, 'logs'],
    queryFn: () => getRunLogs(runId),
    enabled: expanded,
    refetchInterval: expanded ? pollInterval : false,
  })

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
          {temporalError && (
            <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d] mb-3">
              Failed to load workflow details
            </p>
          )}
          {temporalData != null && <TemporalSection details={temporalData} />}
          {logsData != null && <LogsSection logsData={logsData} />}
        </div>
      )}
    </div>
  )
}
