import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { Pencil, Play, Plus, Trash2, Search } from 'lucide-react'
import { deleteModel, listModels } from '@/api/models'
import { RunModal } from '@/components/RunModal'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import type { TrainingModel } from '@/types'

function LoadingState() {
  return (
    <div className="ed-page">
      <div className="flex items-center gap-3">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
          Loading models
        </span>
        <div className="h-px w-24 bg-[#1a1a1a] animate-pulse" />
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="border border-dashed border-[#d0d0c8] bg-white/40 rounded-[4px] py-16 text-center">
      <p className="font-['DM_Serif_Display'] italic text-[1.4rem] text-[#3a3a36] mb-2">
        No models configured.
      </p>
      <p className="font-['Outfit'] text-[0.9rem] text-[#888888] mb-6 max-w-md mx-auto">
        Configure a base model, training parameters, and a backend to begin your first experiment.
      </p>
      <Button asChild>
        <Link to="/models/new">
          <Plus className="h-3.5 w-3.5" />
          Create model
        </Link>
      </Button>
    </div>
  )
}

export function ModelsListPage() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [runTarget, setRunTarget] = useState<TrainingModel | null>(null)

  const { data: models = [], isLoading } = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['models'] }),
  })

  const filtered = models.filter(m => {
    const q = search.toLowerCase()
    return (
      m.name.toLowerCase().includes(q) ||
      m.description.toLowerCase().includes(q) ||
      m.base_model.toLowerCase().includes(q)
    )
  })

  if (isLoading) return <LoadingState />

  return (
    <div className="ed-page">
      {/* Page header — serif title, mono kicker, full-bleed rule */}
      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888] mb-3">
          Vol. 1 · Catalog
        </div>
        <div className="flex items-end justify-between gap-6 flex-wrap">
          <div className="max-w-2xl">
            <h1 className="font-['DM_Serif_Display'] text-[2.4rem] leading-[1.05] text-[#1a1a1a] mb-3">
              Models
            </h1>
            <p className="font-['Outfit'] text-[1rem] text-[#3a3a36] leading-relaxed">
              Select a model to configure training and inference. Each model defines a base architecture,
              hyperparameters, and the linked datasets used across runs.
            </p>
          </div>
          <Button asChild size="lg">
            <Link to="/models/new">
              <Plus className="h-4 w-4" />
              New model
            </Link>
          </Button>
        </div>
        <hr className="ed-rule mt-8 mb-0" />
      </header>

      {models.length === 0 ? (
        <EmptyState />
      ) : (
        <>
          {/* Search bar */}
          <div className="flex items-center gap-3 mb-6">
            <Search className="h-4 w-4 text-[#888888]" />
            <Input
              className="max-w-sm"
              placeholder="Filter by name, description, or base model"
              value={search}
              onChange={e => setSearch(e.target.value)}
              aria-label="Search models"
            />
            <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#888888] ml-auto">
              {filtered.length} / {models.length} shown
            </span>
          </div>

          {/* Editorial table */}
          <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
            <table className="ed-table">
              <thead>
                <tr>
                  <th style={{ width: '4rem' }}>№</th>
                  <th>Model</th>
                  <th>Base</th>
                  <th>Backend</th>
                  <th style={{ width: '5rem' }}>Epochs</th>
                  <th>Status</th>
                  <th style={{ width: '14rem' }}></th>
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="text-center py-10">
                      <span className="font-['DM_Serif_Display'] italic text-[#888888]">
                        No models match "{search}"
                      </span>
                    </td>
                  </tr>
                ) : (
                  filtered.map((model, i) => (
                    <tr
                      key={model.id}
                      className="cursor-pointer"
                      onClick={() => navigate(`/models/${model.id}`)}
                    >
                      <td>
                        <span className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#888888]">
                          {String(i + 1).padStart(2, '0')}
                        </span>
                      </td>
                      <td>
                        <div className="font-['DM_Serif_Display'] text-[1.05rem] text-[#1a1a1a] leading-tight">
                          {model.name}
                        </div>
                        {model.description && (
                          <div className="font-['Outfit'] text-[0.78rem] text-[#888888] mt-0.5 line-clamp-1 max-w-md">
                            {model.description}
                          </div>
                        )}
                      </td>
                      <td className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#3a3a36]">
                        {model.base_model}
                      </td>
                      <td className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#3a3a36]">
                        {model.remote_backend}
                      </td>
                      <td className="font-['IBM_Plex_Mono'] text-[0.85rem] text-[#1a1a1a]">
                        {model.epochs}
                      </td>
                      <td>
                        {model.is_active ? (
                          <RunStatusBadge status="completed" />
                        ) : (
                          <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#b3b1a6]">
                            —
                          </span>
                        )}
                      </td>
                      <td onClick={e => e.stopPropagation()}>
                        <div className="flex gap-2 justify-end">
                          <Button
                            size="sm"
                            onClick={() => setRunTarget(model)}
                            aria-label={`Trigger run for ${model.name}`}
                          >
                            <Play className="h-3 w-3" />Run
                          </Button>
                          <Button size="sm" variant="outline" asChild>
                            <Link
                              to={`/models/${model.id}/edit`}
                              aria-label={`Edit ${model.name}`}
                            >
                              <Pencil className="h-3 w-3" />Edit
                            </Link>
                          </Button>
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => deleteMutation.mutate(model.id)}
                            disabled={
                              deleteMutation.isPending &&
                              deleteMutation.variables === model.id
                            }
                            aria-label={`Delete ${model.name}`}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {runTarget && (
        <RunModal model={runTarget} onClose={() => setRunTarget(null)} />
      )}
    </div>
  )
}
