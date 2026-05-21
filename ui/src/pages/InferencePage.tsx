import { useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, Square, Trash2 } from 'lucide-react'
import { deleteInference, listInferences, startInference, stopInference } from '../api/inferences'
import { InferenceStatusBadge } from '../components/InferenceStatusBadge'
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

  const { data: instances = [], isLoading, isError } = useQuery({
    queryKey: ['inferences'],
    queryFn: listInferences,
  })

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

  if (isLoading) return <p className="p-8 text-gray-500">Loading…</p>
  if (isError) return <div className="p-8 text-red-500">Failed to load inference instances.</div>

  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold mb-6">Inference Instances</h1>

      {instances.length === 0 ? (
        <div className="text-center py-16 text-gray-500">
          <p>No inference instances. Train a model to get started.</p>
        </div>
      ) : (
        <div className="rounded-md border bg-white overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-gray-50 text-gray-500 text-xs uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-semibold">Model ID</th>
                <th className="text-left px-4 py-3 font-semibold">Pod Name</th>
                <th className="text-left px-4 py-3 font-semibold">Status</th>
                <th className="text-left px-4 py-3 font-semibold">Last Used</th>
                <th className="text-left px-4 py-3 font-semibold">Timeout (min)</th>
                <th className="text-left px-4 py-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {instances.map((instance: InferenceInstance) => (
                <tr key={instance.id} className="border-b last:border-0 hover:bg-gray-50">
                  <td className="px-4 py-3 font-mono text-gray-700 text-xs">{instance.model_id}</td>
                  <td className="px-4 py-3 text-gray-700">{instance.pod_name || '—'}</td>
                  <td className="px-4 py-3"><InferenceStatusBadge status={instance.status} /></td>
                  <td className="px-4 py-3 text-gray-500 text-xs">{formatDate(instance.last_used_at)}</td>
                  <td className="px-4 py-3 text-gray-700">{instance.idle_timeout_minutes}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      {CAN_START.includes(instance.status) && (
                        <Button size="sm" onClick={() => startMutation.mutate(instance.id)}
                          aria-label={`Start ${instance.id}`}>
                          <Play className="h-3.5 w-3.5 mr-1" />Start
                        </Button>
                      )}
                      {CAN_STOP.includes(instance.status) && (
                        <Button size="sm" variant="outline" onClick={() => stopMutation.mutate(instance.id)}
                          aria-label={`Stop ${instance.id}`}>
                          <Square className="h-3.5 w-3.5 mr-1" />Stop
                        </Button>
                      )}
                      {CAN_DELETE.includes(instance.status) && (
                        <Button size="sm" variant="destructive" onClick={() => deleteMutation.mutate(instance.id)}
                          aria-label={`Delete ${instance.id}`}>
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
