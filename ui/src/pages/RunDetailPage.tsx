import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { cancelRun, deleteRun, getRunEvaluation, getRun, isRunActive, isRunCancellable } from '@/api/runs'
import { listDatasets } from '@/api/datasets'
import { listInferences } from '@/api/inferences'
import { InferenceStatusBadge } from '@/components/InferenceStatusBadge'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import { PipelineStages } from '@/components/PipelineStages'
import { RunDetailsPanel } from '@/components/RunDetailsPanel'
import { EvalMetrics } from '@/components/EvalMetrics'
import { Button } from '@/components/ui/button'
import type { PipelineStage, StageStatus } from '@/components/PipelineStages'
import type { RunStatus } from '@/types'

function buildStages(status: RunStatus): PipelineStage[] {
  const stageNames = ['Generate', 'Train', 'Evaluate', 'Export']
  const activeMap: Partial<Record<RunStatus, number>> = {
    generating: 0,
    training:   1,
    evaluating: 2,
    exporting:  3,
  }

  if (status === 'completed') {
    return stageNames.map(name => ({ name, status: 'completed' as StageStatus }))
  }
  if (status === 'failed') {
    return stageNames.map((name, i): PipelineStage => ({
      name,
      status: i === 0 ? 'failed' : 'pending',
    }))
  }

  const activeIdx = activeMap[status] ?? -1
  return stageNames.map((name, i): PipelineStage => ({
    name,
    status: i < activeIdx ? 'completed' : i === activeIdx ? 'active' : 'pending',
  }))
}

const EVAL_STATUSES: RunStatus[] = ['completed', 'failed']
const EVAL_PASS_THRESHOLD = 0.95

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: datasetsData } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => listDatasets(),
  })
  const datasetById = Object.fromEntries((datasetsData?.items ?? []).map(d => [d.id, d.name]))

  const { data: run, isLoading } = useQuery({
    queryKey: ['runs', runId],
    queryFn: () => getRun(runId!),
    refetchInterval: (query) => {
      const data = query.state.data
      return data && isRunActive(data) ? 5000 : false
    },
  })

  const showEval = run != null && EVAL_STATUSES.includes(run.status) && run.eval_valid_pct != null

  const { data: evalData, isError: evalError } = useQuery({
    queryKey: ['runs', runId, 'evaluation'],
    queryFn: () => getRunEvaluation(runId!),
    enabled: showEval,
  })

  const { data: instancesData } = useQuery({
    queryKey: ['inferences', { modelId: run?.model_id }],
    queryFn: () => listInferences(1, 50, run!.model_id),
    enabled: run != null,
  })
  const instances = instancesData?.items ?? []

  const deleteMutation = useMutation({
    mutationFn: () => deleteRun(runId!),
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ['runs', runId] })
      navigate('/runs')
    },
  })

  const cancelMutation = useMutation({
    mutationFn: () => cancelRun(runId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs', runId] })
    },
  })

  function handleDelete() {
    if (window.confirm('Delete this run? This cannot be undone.')) {
      deleteMutation.mutate()
    }
  }

  function handleCancel() {
    if (window.confirm('Cancel this run?')) {
      cancelMutation.mutate()
    }
  }

  if (isLoading) {
    return (
      <div className="ed-page">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#6b6b6b]">
          Loading run
        </span>
      </div>
    )
  }
  if (!run) {
    return (
      <div className="ed-page">
        <p className="font-['DM_Serif_Display'] italic text-[#7f1d1d] text-[1.4rem]">
          Run not found.
        </p>
      </div>
    )
  }

  return (
    <div className="ed-page max-w-4xl">
      <Link
        to="/runs"
        className="inline-flex items-center gap-1.5 font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#6b6b6b] hover:text-[#1a1a1a] mb-3"
      >
        <ArrowLeft className="h-3 w-3" />
        Back to runs
      </Link>

      <header className="mb-4">
        <div className="flex items-center justify-between gap-3 mb-2 flex-wrap">
          <div className="flex items-center gap-3 min-w-0">
            <span className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#6b6b6b] shrink-0">
              Run · {run.id.slice(0, 8)}
            </span>
            <RunStatusBadge status={run.status} />
          </div>
          <div className="flex gap-2 shrink-0">
            {isRunCancellable(run) && (
              <Button variant="outline" onClick={handleCancel} disabled={cancelMutation.isPending}>
                {cancelMutation.isPending ? 'Cancelling' : 'Cancel run'}
              </Button>
            )}
            <Button variant="destructive" onClick={handleDelete} disabled={deleteMutation.isPending}>
              {deleteMutation.isPending ? 'Deleting' : 'Delete run'}
            </Button>
          </div>
        </div>
        <h1 className="font-['DM_Serif_Display'] text-[1.75rem] leading-tight text-[#1a1a1a] break-all">
          {run.workflow_id}
        </h1>
      </header>

      {(cancelMutation.isError || deleteMutation.isError) && (
        <div className="mb-6 border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-4 py-3">
          <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
            {cancelMutation.isError ? 'Failed to cancel run. Please try again.' : 'Failed to delete run. Please try again.'}
          </p>
        </div>
      )}

      {/* Pipeline */}
      <section className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] px-5 py-4 mb-6">
        <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#6b6b6b] mb-3">
          Pipeline
        </div>
        <PipelineStages stages={buildStages(run.status)} />
        <RunDetailsPanel runId={runId!} run={run} datasetById={datasetById} />
      </section>

      <hr className="ed-rule" />
      <section className="mb-10">
        <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a] mb-4">Inference instances</h2>
        {instances.length === 0 ? (
          <p className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#6b6b6b] italic">
            No inference instances for this model.
          </p>
        ) : (
          <ol className="flex flex-col">
            {instances.map(inst => (
              <li key={inst.id}>
                <Link
                  to={`/inferences/${inst.id}`}
                  className="flex items-center gap-4 px-4 py-3 border-t border-[#d0d0c8] last:border-b hover:bg-[#f3f2ec] transition-colors"
                >
                  <InferenceStatusBadge status={inst.status} />
                  <span className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">
                    {inst.pod_name || inst.id.slice(0, 8)}
                  </span>
                  <span className="font-['IBM_Plex_Mono'] text-[0.65rem] text-[#6b6b6b] ml-auto">
                    {inst.id.slice(0, 8)}
                  </span>
                </Link>
              </li>
            ))}
          </ol>
        )}
      </section>

      {showEval && run.eval_valid_pct != null && (
        <>
          <hr className="ed-rule" />
          <section>
            <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a] mb-4">
              Evaluation results
            </h2>
            {evalError && (
              <div className="mb-4 border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-3 py-2">
                <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
                  Failed to load detailed report.
                </p>
              </div>
            )}
            <EvalMetrics
              validPct={run.eval_valid_pct}
              passed={evalData?.quality_report?.passed ?? (run.eval_valid_pct >= EVAL_PASS_THRESHOLD)}
              qualityReport={evalData?.quality_report}
            />
          </section>
        </>
      )}
    </div>
  )
}
