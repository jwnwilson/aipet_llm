import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, ChevronUp, Play, Plus, Square, Trash2 } from 'lucide-react'
import { createInference, deleteInference, listInferences, startInference, stopInference } from '@/api/inferences'
import { listModels } from '@/api/models'
import { InferenceStatusBadge } from '@/components/InferenceStatusBadge'
import { InstanceInferencePanel } from '@/components/InstanceInferencePanel'
import { Pagination } from '@/components/Pagination'
import { Button } from '@/components/ui/button'
import type { InferenceInstance, InferenceStatus, TrainingModel } from '@/types'

const AUTO_REFRESH_MS = 10_000
const CAN_START: InferenceStatus[] = ['pending', 'shutdown', 'failed']
const CAN_STOP: InferenceStatus[] = ['available', 'initializing', 'idle']
const CAN_DELETE: InferenceStatus[] = ['pending', 'shutdown', 'failed']

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

interface ModelGroupProps {
  model: TrainingModel | null
  instances: InferenceInstance[]
  onStart: (id: string) => void
  onStop: (id: string) => void
  onDelete: (id: string) => void
  pendingStart: string | null
  pendingStop: string | null
  pendingDelete: string | null
}

function ModelGroup({
  model, instances, onStart, onStop, onDelete,
  pendingStart, pendingStop, pendingDelete,
}: ModelGroupProps) {
  const [expandedTest, setExpandedTest] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const addMutation = useMutation({
    mutationFn: () => createInference({ model_id: model?.id ?? instances[0]?.model_id ?? '' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })

  return (
    <section className="mb-8">
      <div className="flex items-center gap-3 mb-3">
        <h2 className="font-['DM_Serif_Display'] text-[1.3rem] text-[#1a1a1a]">
          {model?.name ?? 'Unknown model'}
        </h2>
        <span className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888]">
          {instances.length} instance{instances.length !== 1 ? 's' : ''}
        </span>
        <Button
          size="sm"
          variant="outline"
          onClick={() => addMutation.mutate()}
          disabled={addMutation.isPending}
          aria-label="Add instance"
        >
          <Plus className="h-3 w-3" />
          {addMutation.isPending ? 'Adding…' : 'Add instance'}
        </Button>
      </div>

      <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
        {instances.map((instance, i) => (
          <div key={instance.id} className={i < instances.length - 1 ? 'border-b border-[#e5e3d8]' : ''}>
            <div className="flex items-center justify-between gap-4 px-5 py-4 flex-wrap">
              <div className="flex items-center gap-6 min-w-0">
                <InferenceStatusBadge status={instance.status} />
                <div>
                  <p className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">
                    {instance.pod_name || '—'}
                  </p>
                  <p className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#888888] mt-0.5">
                    Last used: {formatDate(instance.last_used_at)}
                  </p>
                </div>
              </div>

              <div className="flex gap-2 items-center shrink-0">
                {instance.status === 'available' && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setExpandedTest(prev => prev === instance.id ? null : instance.id)}
                    aria-label="Test inference"
                  >
                    {expandedTest === instance.id
                      ? <ChevronUp className="h-3 w-3" />
                      : <ChevronDown className="h-3 w-3" />}
                    Test
                  </Button>
                )}
                {CAN_START.includes(instance.status) && (
                  <Button size="sm" onClick={() => onStart(instance.id)}
                    disabled={pendingStart === instance.id}
                    aria-label={`Start ${instance.id}`}>
                    <Play className="h-3 w-3" />Start
                  </Button>
                )}
                {CAN_STOP.includes(instance.status) && (
                  <Button size="sm" variant="outline" onClick={() => onStop(instance.id)}
                    disabled={pendingStop === instance.id}
                    aria-label={`Stop ${instance.id}`}>
                    <Square className="h-3 w-3" />Stop
                  </Button>
                )}
                {CAN_DELETE.includes(instance.status) && (
                  <Button size="sm" variant="destructive" onClick={() => onDelete(instance.id)}
                    disabled={pendingDelete === instance.id}
                    aria-label={`Delete ${instance.id}`}>
                    <Trash2 className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>

            {expandedTest === instance.id && (
              <div className="px-5 pb-4">
                <InstanceInferencePanel instanceId={instance.id} />
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

export function InferencePage() {
  const queryClient = useQueryClient()

  const [page, setPage] = useState(1)
  const { data: instancesData, isLoading, isError } = useQuery({
    queryKey: ['inferences', page],
    queryFn: () => listInferences(page),
  })
  const instances: InferenceInstance[] = instancesData?.items ?? []

  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: () => listModels(),
  })
  const modelsList: TrainingModel[] = modelsData?.items ?? []
  const modelsById = new Map<string, TrainingModel>(modelsList.map(m => [m.id, m]))

  useEffect(() => {
    const interval = setInterval(
      () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
      AUTO_REFRESH_MS,
    )
    return () => clearInterval(interval)
  }, [queryClient])

  const startMutation = useMutation({
    mutationFn: (id: string) => startInference(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })
  const stopMutation = useMutation({
    mutationFn: (id: string) => stopInference(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteInference(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })

  const grouped = new Map<string, InferenceInstance[]>()
  for (const inst of instances) {
    const list = grouped.get(inst.model_id) ?? []
    grouped.set(inst.model_id, [...list, inst])
  }

  if (isLoading) {
    return (
      <div className="ed-page">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
          Loading instances
        </span>
      </div>
    )
  }
  if (isError) {
    return (
      <div className="ed-page">
        <div className="border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-4 py-3 inline-block">
          <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
            Failed to load inference instances.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="ed-page">
      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888] mb-3">
          Vol. 4 · Runtime
        </div>
        <h1 className="font-['DM_Serif_Display'] text-[2.4rem] leading-[1.05] text-[#1a1a1a] mb-3">
          Inference instances
        </h1>
        <p className="font-['Outfit'] text-[1rem] text-[#3a3a36] max-w-2xl leading-relaxed">
          Inference pods grouped by model. Start an instance to make a model available for serving;
          use the Test panel to run a sample request against any available pod.
        </p>
        <hr className="ed-rule mt-7 mb-0" />
      </header>

      {grouped.size === 0 ? (
        <div className="border border-dashed border-[#d0d0c8] bg-white/40 rounded-[4px] py-16 text-center">
          <p className="font-['DM_Serif_Display'] italic text-[1.4rem] text-[#3a3a36] mb-1">
            No inference instances.
          </p>
          <p className="font-['Outfit'] text-[0.9rem] text-[#888888]">
            Train a model to provision a serving instance.
          </p>
        </div>
      ) : (
        <>
          {Array.from(grouped.entries()).map(([modelId, modelInstances]) => (
            <ModelGroup
              key={modelId}
              model={modelsById.get(modelId) ?? null}
              instances={modelInstances}
              onStart={id => startMutation.mutate(id)}
              onStop={id => stopMutation.mutate(id)}
              onDelete={id => deleteMutation.mutate(id)}
              pendingStart={startMutation.isPending ? (startMutation.variables ?? null) : null}
              pendingStop={stopMutation.isPending ? (stopMutation.variables ?? null) : null}
              pendingDelete={deleteMutation.isPending ? (deleteMutation.variables ?? null) : null}
            />
          ))}
          <Pagination page={page} pages={instancesData?.pages ?? 1} onPageChange={setPage} />
        </>
      )}
    </div>
  )
}
