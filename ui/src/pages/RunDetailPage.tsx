import React from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { cancelRun, deleteRun, getRunEvaluation, getRun, isRunActive, isRunCancellable } from '@/api/runs'
import { listDatasets } from '@/api/datasets'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import { PipelineStages } from '@/components/PipelineStages'
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

function MetricBlock({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 py-3 border-b border-[#e5e3d8]">
      <dt className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888]">
        {label}
      </dt>
      <dd className="font-['IBM_Plex_Mono'] text-[0.88rem] text-[#1a1a1a] break-all">
        {value}
      </dd>
    </div>
  )
}

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets'],
    queryFn: listDatasets,
  })
  const datasetById = Object.fromEntries(datasets.map(d => [d.id, d.name]))

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
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
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
        className="inline-flex items-center gap-1.5 font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#888888] hover:text-[#1a1a1a] mb-5"
      >
        <ArrowLeft className="h-3 w-3" />
        Back to runs
      </Link>

      <header className="mb-8">
        <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#888888] mb-2">
          Run · {run.id.slice(0, 8)}
        </div>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="min-w-0">
            <h1 className="font-['DM_Serif_Display'] text-[2rem] leading-tight text-[#1a1a1a] mb-3 break-all">
              {run.workflow_id}
            </h1>
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
      </header>

      {(cancelMutation.isError || deleteMutation.isError) && (
        <div className="mb-6 border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-4 py-3">
          <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
            {cancelMutation.isError ? 'Failed to cancel run. Please try again.' : 'Failed to delete run. Please try again.'}
          </p>
        </div>
      )}

      {/* Pipeline */}
      <section className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] px-8 py-6 mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#888888] mb-5">
          Pipeline stages
        </div>
        <PipelineStages stages={buildStages(run.status)} />
      </section>

      <hr className="ed-rule" />

      {/* Metrics grid */}
      <section className="mb-10">
        <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a] mb-4">Metrics</h2>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-10">
          <MetricBlock label="Run ID" value={run.id} />
          <MetricBlock label="Started" value={new Date(run.created_at).toLocaleString()} />
          <MetricBlock label="Updated" value={new Date(run.updated_at).toLocaleString()} />
          {run.progress != null && (
            <MetricBlock label="Progress" value={`${Math.round(run.progress * 100)}%`} />
          )}
          {run.eval_valid_pct != null && (
            <MetricBlock
              label="Eval valid"
              value={`${Math.round(run.eval_valid_pct * 100)}%`}
            />
          )}
          {run.progress_detail && (
            <MetricBlock label="Detail" value={run.progress_detail} />
          )}
          {run.train_dataset_id != null && (
            <MetricBlock
              label="Train dataset"
              value={datasetById[run.train_dataset_id] ?? run.train_dataset_id}
            />
          )}
          {run.eval_dataset_id != null && (
            <MetricBlock
              label="Eval dataset"
              value={datasetById[run.eval_dataset_id] ?? run.eval_dataset_id}
            />
          )}
        </dl>
      </section>

      {run.training_config && Object.keys(run.training_config).length > 0 && (
        <>
          <hr className="ed-rule" />
          <section className="mb-10">
            <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a] mb-4">
              Run configuration
            </h2>
            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-10">
              {Object.entries(run.training_config)
                .filter(([, v]) => v != null)
                .map(([k, v]) => (
                  <React.Fragment key={k}>
                    <MetricBlock label={k.replace(/_/g, ' ')} value={String(v)} />
                  </React.Fragment>
                ))}
            </dl>
          </section>
        </>
      )}

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
