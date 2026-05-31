import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, Square, Trash2 } from 'lucide-react'
import { deleteInference, listInferences, startInference, stopInference } from '../api/inferences'
import { InferenceStatusBadge } from '../components/InferenceStatusBadge'
import { Pagination } from '../components/Pagination'
import { Button } from '../components/ui/button'
import type { InferenceInstance, InferenceStatus } from '../types'

const AUTO_REFRESH_MS = 10_000
const CAN_START: InferenceStatus[] = ['pending', 'shutdown', 'failed']
const CAN_STOP: InferenceStatus[] = ['available', 'initializing', 'idle']
const CAN_DELETE: InferenceStatus[] = ['pending', 'shutdown', 'failed']

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

export function InferencePage() {
  const queryClient = useQueryClient()

  const [page, setPage] = useState(1)
  const { data: instancesData, isLoading, isError } = useQuery({
    queryKey: ['inferences', page],
    queryFn: () => listInferences(page),
  })
  const instances = instancesData?.items ?? []

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
          Inference pods host trained models for serving. Start an instance to make a model available
          for live inference; stop it to release compute.
        </p>
        <hr className="ed-rule mt-7 mb-0" />
      </header>

      {instances.length === 0 ? (
        <div className="border border-dashed border-[#d0d0c8] bg-white/40 rounded-[4px] py-16 text-center">
          <p className="font-['DM_Serif_Display'] italic text-[1.4rem] text-[#3a3a36] mb-1">
            No inference instances.
          </p>
          <p className="font-['Outfit'] text-[0.9rem] text-[#888888]">
            Train a model to provision a serving instance.
          </p>
        </div>
      ) : (
        <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
          <table className="ed-table">
            <thead>
              <tr>
                <th>Model ID</th>
                <th>Pod name</th>
                <th>Status</th>
                <th>Last used</th>
                <th>Timeout (min)</th>
                <th style={{ width: '18rem' }}></th>
              </tr>
            </thead>
            <tbody>
              {instances.map((instance: InferenceInstance) => (
                <tr key={instance.id}>
                  <td className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#1a1a1a]">
                    {instance.model_id}
                  </td>
                  <td className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#3a3a36]">
                    {instance.pod_name || '—'}
                  </td>
                  <td><InferenceStatusBadge status={instance.status} /></td>
                  <td className="font-['IBM_Plex_Mono'] text-[0.74rem] text-[#888888]">
                    {formatDate(instance.last_used_at)}
                  </td>
                  <td className="font-['IBM_Plex_Mono'] text-[0.85rem] text-[#1a1a1a]">
                    {instance.idle_timeout_minutes}
                  </td>
                  <td>
                    <div className="flex gap-2 justify-end">
                      {CAN_START.includes(instance.status) && (
                        <Button size="sm" onClick={() => startMutation.mutate(instance.id)}
                          aria-label={`Start ${instance.id}`}>
                          <Play className="h-3 w-3" />Start
                        </Button>
                      )}
                      {CAN_STOP.includes(instance.status) && (
                        <Button size="sm" variant="outline" onClick={() => stopMutation.mutate(instance.id)}
                          aria-label={`Stop ${instance.id}`}>
                          <Square className="h-3 w-3" />Stop
                        </Button>
                      )}
                      {CAN_DELETE.includes(instance.status) && (
                        <Button size="sm" variant="destructive" onClick={() => deleteMutation.mutate(instance.id)}
                          aria-label={`Delete ${instance.id}`}>
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pagination page={page} pages={instancesData?.pages ?? 1} onPageChange={setPage} />
      )}
    </div>
  )
}
