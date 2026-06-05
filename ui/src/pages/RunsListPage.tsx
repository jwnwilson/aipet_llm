import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { listRuns } from '@/api/runs'
import { listModels } from '@/api/models'
import { Pagination } from '@/components/Pagination'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import type { RunRecord, TrainingModel } from '@/types'

interface RunModelGroupProps {
  model: TrainingModel | null
  runs: RunRecord[]
}

function RunModelGroup({ model, runs }: RunModelGroupProps) {
  return (
    <section className="mb-8">
      <div className="flex items-baseline gap-3 mb-3">
        <h2 className="font-['DM_Serif_Display'] text-[1.3rem] text-[#1a1a1a]">
          {model?.name ?? 'Unknown model'}
        </h2>
        <span className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
          {runs.length} run{runs.length !== 1 ? 's' : ''}
        </span>
      </div>

      <ol className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
        {runs.map((run, i) => (
          <li key={run.id}>
            <Link
              to={`/runs/${run.id}`}
              aria-label={run.workflow_id}
              className="flex items-center justify-between gap-5 px-6 py-4 border-b border-[#e5e3d8] last:border-b-0 hover:bg-[#f3f2ec] transition-colors"
            >
              <div className="flex items-baseline gap-4 min-w-0">
                <span className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#6b6b6b] shrink-0">
                  {String(runs.length - i).padStart(3, '0')}
                </span>
                <div className="min-w-0">
                  <p className="font-['IBM_Plex_Mono'] text-[0.85rem] text-[#1a1a1a] truncate">
                    {run.workflow_id}
                  </p>
                  <p className="font-['Outfit'] text-[0.78rem] text-[#6b6b6b] mt-0.5">
                    {new Date(run.created_at).toLocaleString()}
                  </p>
                </div>
              </div>
              <RunStatusBadge status={run.status} />
            </Link>
          </li>
        ))}
      </ol>
    </section>
  )
}

export function RunsListPage() {
  const [page, setPage] = useState(1)

  const { data: runsData, isLoading, isError } = useQuery({
    queryKey: ['runs', page],
    queryFn: () => listRuns(page),
  })
  const runs = runsData?.items ?? []

  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: () => listModels(),
  })
  const modelsById = new Map<string, TrainingModel>(
    (modelsData?.items ?? []).map(m => [m.id, m])
  )

  const grouped = new Map<string, RunRecord[]>()
  for (const run of runs) {
    const list = grouped.get(run.model_id) ?? []
    grouped.set(run.model_id, [...list, run])
  }

  if (isLoading) {
    return (
      <div className="ed-page">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#6b6b6b]">
          Loading runs
        </span>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="ed-page">
        <div className="border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-4 py-3 inline-block">
          <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
            Failed to load training runs.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="ed-page">
      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#6b6b6b] mb-3">
          Vol. 2 · History
        </div>
        <h1 className="font-['DM_Serif_Display'] text-[2.4rem] leading-[1.05] text-[#1a1a1a] mb-2">
          Training runs
        </h1>
        <p className="font-['Outfit'] text-[0.95rem] text-[#3a3a36] max-w-2xl">
          Every training run is logged with its workflow id, status, and timing. Click into a run
          to inspect pipeline stages, configuration, and evaluation results.
        </p>
        <hr className="ed-rule mt-7 mb-0" />
      </header>

      {grouped.size === 0 ? (
        <div className="border border-dashed border-[#d0d0c8] bg-white/40 rounded-[4px] py-16 text-center">
          <p className="font-['DM_Serif_Display'] italic text-[1.4rem] text-[#3a3a36] mb-1">
            No runs yet.
          </p>
          <p className="font-['Outfit'] text-[0.9rem] text-[#6b6b6b]">
            Trigger a run from a model's detail page.
          </p>
        </div>
      ) : (
        <>
          {Array.from(grouped.entries()).map(([modelId, modelRuns]) => (
            <RunModelGroup
              key={modelId}
              model={modelsById.get(modelId) ?? null}
              runs={modelRuns}
            />
          ))}
          <Pagination page={page} pages={runsData?.pages ?? 1} onPageChange={setPage} />
        </>
      )}
    </div>
  )
}
