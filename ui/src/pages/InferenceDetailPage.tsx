import React from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Play, Square, Trash2 } from 'lucide-react'
import { deleteInference, getInference, startInference, stopInference, updateInference } from '@/api/inferences'
import { getModel } from '@/api/models'
import { InferenceStatusBadge } from '@/components/InferenceStatusBadge'
import { InstanceInferencePanel } from '@/components/InstanceInferencePanel'
import { Button } from '@/components/ui/button'
import type { InferenceStatus } from '@/types'

const CAN_START: InferenceStatus[] = ['pending', 'shutdown', 'failed']
const CAN_STOP: InferenceStatus[] = ['available', 'initializing', 'idle']
const CAN_DELETE: InferenceStatus[] = ['pending', 'shutdown', 'failed']
const TRANSITIONING: InferenceStatus[] = ['pending', 'initializing']

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

function MetricBlock({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 py-3 border-b border-[#e5e3d8]">
      <dt className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
        {label}
      </dt>
      <dd className="font-['IBM_Plex_Mono'] text-[0.88rem] text-[#1a1a1a] break-all">
        {value}
      </dd>
    </div>
  )
}

export function InferenceDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: instance, isLoading } = useQuery({
    queryKey: ['inferences', id],
    queryFn: () => getInference(id!),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status && TRANSITIONING.includes(status) ? 10_000 : false
    },
  })

  const { data: model } = useQuery({
    queryKey: ['models', instance?.model_id],
    queryFn: () => getModel(instance!.model_id),
    enabled: instance != null,
  })

  const startMutation = useMutation({
    mutationFn: () => startInference(id!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences', id] }),
  })

  const stopMutation = useMutation({
    mutationFn: () => stopInference(id!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences', id] }),
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteInference(id!),
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ['inferences', id] })
      navigate('/inferences')
    },
  })

  const keepAliveMutation = useMutation({
    mutationFn: (keepAlive: boolean) => updateInference(id!, { keep_alive: keepAlive }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences', id] }),
  })

  if (isLoading) {
    return (
      <div className="ed-page">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#6b6b6b]">
          Loading instance
        </span>
      </div>
    )
  }
  if (!instance) {
    return (
      <div className="ed-page">
        <p className="font-['DM_Serif_Display'] italic text-[#7f1d1d] text-[1.4rem]">
          Instance not found.
        </p>
      </div>
    )
  }

  return (
    <div className="ed-page max-w-4xl">
      <Link
        to="/inferences"
        className="inline-flex items-center gap-1.5 font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#6b6b6b] hover:text-[#1a1a1a] mb-5"
      >
        <ArrowLeft className="h-3 w-3" />
        Back to instances
      </Link>

      <header className="mb-8">
        <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#6b6b6b] mb-2">
          Inference · {instance.id.slice(0, 8)}
        </div>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="min-w-0">
            <h1 className="font-['DM_Serif_Display'] text-[2rem] leading-tight text-[#1a1a1a] mb-3 break-all">
              {instance.pod_name || instance.id}
            </h1>
            <InferenceStatusBadge status={instance.status} />
          </div>
          <div className="flex gap-2 shrink-0">
            {CAN_START.includes(instance.status) && (
              <Button onClick={() => startMutation.mutate()} disabled={startMutation.isPending}>
                <Play className="h-3.5 w-3.5" />
                {startMutation.isPending ? 'Starting…' : 'Start'}
              </Button>
            )}
            {CAN_STOP.includes(instance.status) && (
              <Button variant="outline" onClick={() => stopMutation.mutate()} disabled={stopMutation.isPending}>
                <Square className="h-3.5 w-3.5" />
                {stopMutation.isPending ? 'Stopping…' : 'Stop'}
              </Button>
            )}
            {CAN_DELETE.includes(instance.status) && (
              <Button
                variant="destructive"
                size="icon"
                onClick={() => deleteMutation.mutate()}
                disabled={deleteMutation.isPending}
                aria-label="Delete instance"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        </div>
      </header>

      <section className="mb-10">
        <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a] mb-4">Details</h2>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-10">
          <MetricBlock label="Instance ID" value={instance.id} />
          <MetricBlock
            label="Model"
            value={
              model ? (
                <Link to={`/models/${instance.model_id}`} className="hover:underline">
                  {model.name}
                </Link>
              ) : (
                instance.model_id
              )
            }
          />
          <MetricBlock label="Pod name" value={instance.pod_name || '—'} />
          <MetricBlock label="Pod namespace" value={instance.pod_namespace || '—'} />
          <MetricBlock label="Idle timeout" value={`${instance.idle_timeout_minutes} min`} />
          <MetricBlock label="Created" value={formatDate(instance.created_at)} />
          <MetricBlock label="Last used" value={formatDate(instance.last_used_at)} />
          <MetricBlock label="Updated" value={formatDate(instance.updated_at)} />
          <MetricBlock
            label="Keep alive"
            value={
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5 rounded border-[#d0d0c8] accent-[#1a1a1a]"
                  checked={instance.keep_alive}
                  disabled={keepAliveMutation.isPending}
                  onChange={e => keepAliveMutation.mutate(e.target.checked)}
                  aria-label="Keep alive — skip idle shutdown"
                />
                <span className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#3a3a36] select-none">
                  {instance.keep_alive ? 'Enabled — will not be shut down when idle' : 'Disabled — will shut down when idle'}
                </span>
              </label>
            }
          />
        </dl>
      </section>

      {instance.status === 'available' && (
        <>
          <hr className="ed-rule mb-8" />
          <section>
            <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a] mb-4">Test</h2>
            <InstanceInferencePanel instanceId={instance.id} />
          </section>
        </>
      )}
    </div>
  )
}
