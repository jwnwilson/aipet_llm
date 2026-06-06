import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { Pencil, Play, Plus, Trash2, Search } from 'lucide-react'
import { deleteModel, listModels } from '@/api/models'
import { Pagination } from '@/components/Pagination'
import { RunModal } from '@/components/RunModal'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import { useMediaQuery } from '@/hooks/useMediaQuery'
import type { TrainingModel } from '@/types'

function LoadingState() {
  return (
    <div className="ed-page">
      <div className="flex items-center gap-3">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#6b6b6b]">
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
      <p className="font-['Outfit'] text-[0.9rem] text-[#6b6b6b] mb-6 max-w-md mx-auto">
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

function ModelMobileCard({
  model,
  index,
  onRun,
  onDelete,
  deletePending,
}: {
  model: TrainingModel
  index: number
  onRun: () => void
  onDelete: () => void
  deletePending: boolean
}) {
  const navigate = useNavigate()
  return (
    <div
      data-testid="model-mobile-card"
      className="border-b border-[#e5e3d8] px-4 py-4 bg-white last:border-b-0 cursor-pointer"
      onClick={() => navigate(`/models/${model.id}`)}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#6b6b6b]">
              {String(index + 1).padStart(2, '0')}
            </span>
            <span className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#6b6b6b]">
              {model.base_model}
            </span>
          </div>
          <h3 className="font-['DM_Serif_Display'] text-[1.05rem] text-[#1a1a1a] leading-tight">
            {model.name}
          </h3>
          {model.description && (
            <p className="font-['Outfit'] text-[0.78rem] text-[#6b6b6b] mt-0.5 line-clamp-1">
              {model.description}
            </p>
          )}
          <div className="flex items-center gap-3 mt-2 flex-wrap">
            <span className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#3a3a36]">
              {model.remote_backend}
            </span>
            <span className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#6b6b6b]">
              {model.epochs} epochs
            </span>
            {model.is_active && <RunStatusBadge status="completed" />}
          </div>
        </div>
        <div
          className="flex flex-col gap-1.5 shrink-0"
          onClick={e => e.stopPropagation()}
        >
          <Button size="sm" onClick={onRun} aria-label={`Trigger run for ${model.name}`}>
            <Play className="h-3 w-3" />Run
          </Button>
          <Button size="sm" variant="outline" asChild>
            <Link to={`/models/${model.id}/edit`} aria-label={`Edit ${model.name}`}>
              <Pencil className="h-3 w-3" />Edit
            </Link>
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={onDelete}
            disabled={deletePending}
            aria-label={`Delete ${model.name}`}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  )
}

export function ModelsListPage() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [runTarget, setRunTarget] = useState<TrainingModel | null>(null)
  const [page, setPage] = useState(1)

  useEffect(() => { setPage(1) }, [search])

  const { data: modelsData, isLoading } = useQuery({
    queryKey: ['models', page],
    queryFn: () => listModels(page),
  })
  const models = modelsData?.items ?? []

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

  const isMobile = useMediaQuery('(max-width: 767px)')

  if (isLoading) return <LoadingState />

  return (
    <div className="ed-page">
      {/* Page header — serif title, mono kicker, full-bleed rule */}
      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#6b6b6b] mb-3">
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
            <Search className="h-4 w-4 text-[#6b6b6b]" />
            <Input
              className="max-w-sm"
              placeholder="Filter by name, description, or base model"
              value={search}
              onChange={e => setSearch(e.target.value)}
              aria-label="Search models"
            />
            <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#6b6b6b] ml-auto">
              {filtered.length} / {models.length} shown
            </span>
          </div>

          <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
            {isMobile ? (
              filtered.length === 0 ? (
                <div className="px-4 py-10 text-center">
                  <span className="font-['DM_Serif_Display'] italic text-[#6b6b6b]">
                    No models match "{search}"
                  </span>
                </div>
              ) : (
                filtered.map((model, i) => (
                  <ModelMobileCard
                    key={model.id}
                    model={model}
                    index={i}
                    onRun={() => setRunTarget(model)}
                    onDelete={() => deleteMutation.mutate(model.id)}
                    deletePending={
                      deleteMutation.isPending && deleteMutation.variables === model.id
                    }
                  />
                ))
              )
            ) : (
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
                        <span className="font-['DM_Serif_Display'] italic text-[#6b6b6b]">
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
                          <span className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#6b6b6b]">
                            {String(i + 1).padStart(2, '0')}
                          </span>
                        </td>
                        <td>
                          <div className="font-['DM_Serif_Display'] text-[1.05rem] text-[#1a1a1a] leading-tight">
                            {model.name}
                          </div>
                          {model.description && (
                            <div className="font-['Outfit'] text-[0.78rem] text-[#6b6b6b] mt-0.5 line-clamp-1 max-w-md">
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
                            <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#767676]">
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
            )}
          </div>
          <Pagination page={page} pages={modelsData?.pages ?? 1} onPageChange={setPage} />
        </>
      )}

      {runTarget && (
        <RunModal model={runTarget} onClose={() => setRunTarget(null)} />
      )}
    </div>
  )
}
