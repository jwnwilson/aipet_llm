import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useParams, useNavigate } from 'react-router-dom'
import { Play, Pencil, Trash2, ArrowLeft } from 'lucide-react'
import { deleteModel, getModel } from '@/api/models'
import { listRuns, triggerRun } from '@/api/runs'
import { LinkedDatasetsCard } from '@/components/LinkedDatasetsCard'
import { InferencePanel } from '@/components/InferencePanel'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { TrainingModel } from '@/types'

type PipelineStep = {
  num: string
  label: string
  state: 'done' | 'active' | 'pending'
}

function buildPipeline(model: TrainingModel, hasRuns: boolean, hasCompletedRun: boolean): PipelineStep[] {
  const hasDataset = Boolean(model.train_data && model.eval_data)
  const isReady = model.inference_status === 'ready' || model.backend === 'openrouter'

  return [
    { num: '01', label: 'Model',     state: 'done' },
    { num: '02', label: 'Dataset',   state: hasDataset ? 'done' : 'active' },
    { num: '03', label: 'Training',  state: hasCompletedRun ? 'done' : hasRuns ? 'active' : hasDataset ? 'active' : 'pending' },
    { num: '04', label: 'Inference', state: isReady ? 'done' : hasCompletedRun ? 'active' : 'pending' },
  ]
}

function PipelineHeader({ steps }: { steps: PipelineStep[] }) {
  return (
    <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] px-8 py-6">
      <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#888888] mb-4">
        Pipeline
      </div>
      <div className="flex items-center w-full">
        {steps.map((step, i) => {
          const isLast = i === steps.length - 1
          const circleClasses =
            step.state === 'active'
              ? 'bg-[#1a1a1a] text-[#fafaf7] border-[#1a1a1a]'
              : step.state === 'done'
              ? 'bg-[#1a1a1a] text-[#fafaf7] border-[#1a1a1a]'
              : 'bg-white text-[#b3b1a6] border-[#d0d0c8]'
          const labelClasses =
            step.state === 'pending' ? 'text-[#b3b1a6]' : 'text-[#1a1a1a]'
          return (
            <div key={step.label} className={`flex items-center ${isLast ? 'flex-none' : 'flex-1'}`}>
              <div className="flex items-center gap-3 min-w-fit">
                <span
                  className={[
                    'inline-flex items-center justify-center h-10 w-10 rounded-full',
                    "font-['IBM_Plex_Mono'] text-[0.78rem] font-medium",
                    'border-[1.5px] transition-colors',
                    circleClasses,
                  ].join(' ')}
                >
                  {step.num}
                </span>
                <span className={`font-['Outfit'] text-[0.85rem] font-semibold uppercase tracking-[0.12em] ${labelClasses}`}>
                  {step.label}
                </span>
              </div>
              {!isLast && (
                <div
                  aria-hidden
                  className={`flex-1 h-px mx-4 ${
                    steps[i + 1].state !== 'pending' ? 'bg-[#1a1a1a]' : 'bg-[#d0d0c8]'
                  }`}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ConfigTable({ model }: { model: TrainingModel }) {
  const rows: Array<[string, string | number]> = [
    ['Base model', model.base_model],
    ['Training data', model.train_data],
    ['Eval data', model.eval_data],
    ['Epochs', model.epochs],
    ['Patience', model.patience],
    ['Warmup ratio', model.warmup_ratio],
    ['Remote backend', model.remote_backend],
    ['Skip generate', model.skip_generate ? 'Yes' : 'No'],
    ...(model.gguf_path ? [['GGUF path', model.gguf_path] as [string, string]] : []),
  ]
  return (
    <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-0">
      {rows.map(([key, val], i) => (
        <div
          key={String(key)}
          className={`flex flex-col gap-1 py-3 ${i >= 2 ? 'border-t border-[#e5e3d8]' : ''}`}
        >
          <dt className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888]">
            {key}
          </dt>
          <dd className="font-['IBM_Plex_Mono'] text-[0.88rem] text-[#1a1a1a] break-all">
            {String(val)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

export function ModelDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: model, isLoading } = useQuery({
    queryKey: ['models', id],
    queryFn: () => getModel(id!),
  })

  const { data: allRunsData } = useQuery({ queryKey: ['runs'], queryFn: () => listRuns() })
  const runs = (allRunsData?.items ?? []).filter(r => r.model_id === id)
  const hasCompletedRun = runs.some(r => r.status === 'completed')

  const triggerMutation = useMutation({
    mutationFn: () => triggerRun({ model_id: id! }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['runs'] }),
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteModel(id!),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['models'] }); navigate('/models') },
  })

  if (isLoading || !model) {
    return (
      <div className="ed-page">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
          Loading model
        </span>
      </div>
    )
  }

  const pipeline = buildPipeline(model, runs.length > 0, hasCompletedRun)

  return (
    <div className="ed-page">
      {/* Breadcrumb */}
      <Link
        to="/models"
        className="inline-flex items-center gap-1.5 font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#888888] hover:text-[#1a1a1a] mb-5"
      >
        <ArrowLeft className="h-3 w-3" />
        Back to models
      </Link>

      {/* Heading + actions */}
      <header className="mb-8">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div>
            <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#888888] mb-2">
              Model · {model.id.slice(0, 8)}
            </div>
            <h1 className="font-['DM_Serif_Display'] text-[2.6rem] leading-[1.05] text-[#1a1a1a] mb-2">
              {model.name}
            </h1>
            {model.description && (
              <p className="font-['Outfit'] text-[1rem] text-[#3a3a36] max-w-2xl leading-relaxed">
                {model.description}
              </p>
            )}
          </div>
          <div className="flex gap-2 shrink-0">
            <Button onClick={() => triggerMutation.mutate()} disabled={triggerMutation.isPending}>
              <Play className="h-3.5 w-3.5" />
              {triggerMutation.isPending ? 'Starting' : 'Run'}
            </Button>
            <Button variant="outline" asChild>
              <Link to={`/models/${id}/edit`}><Pencil className="h-3.5 w-3.5" />Edit</Link>
            </Button>
            <Button
              variant="destructive"
              size="icon"
              onClick={() => deleteMutation.mutate()}
              disabled={deleteMutation.isPending}
              aria-label="Delete model"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </header>

      {/* Pipeline header */}
      <section className="mb-10">
        <PipelineHeader steps={pipeline} />
      </section>

      <hr className="ed-rule mb-10" />

      {/* Two-column workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        <div className="flex flex-col gap-8">
          <Card>
            <CardHeader>
              <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888]">
                Section I
              </div>
              <CardTitle>Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <ConfigTable model={model} />
            </CardContent>
          </Card>

          <LinkedDatasetsCard model={model} />
        </div>

        <div className="flex flex-col gap-8">
          <InferencePanel model={model} />

          <section>
            <div className="flex items-baseline justify-between mb-4">
              <div>
                <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888]">
                  Section III
                </div>
                <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a]">
                  Run history
                </h2>
              </div>
              <span className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#888888]">
                {runs.length} {runs.length === 1 ? 'run' : 'runs'}
              </span>
            </div>
            {runs.length === 0 ? (
              <div className="border border-dashed border-[#d0d0c8] bg-white/40 rounded-[4px] py-10 text-center">
                <p className="font-['DM_Serif_Display'] italic text-[1.05rem] text-[#888888]">
                  No runs recorded.
                </p>
              </div>
            ) : (
              <ol className="flex flex-col">
                {runs.map((run, i) => (
                  <li key={run.id}>
                    <Link
                      to={`/runs/${run.id}`}
                      className="flex items-center justify-between gap-4 px-4 py-3 border-t border-[#d0d0c8] last:border-b hover:bg-[#f3f2ec] transition-colors"
                    >
                      <div className="flex items-baseline gap-3 min-w-0">
                        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] text-[#888888]">
                          {String(runs.length - i).padStart(2, '0')}
                        </span>
                        <span className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a] truncate">
                          {run.workflow_id}
                        </span>
                      </div>
                      <RunStatusBadge status={run.status} />
                    </Link>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
