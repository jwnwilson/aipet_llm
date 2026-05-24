import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { listRuns } from '@/api/runs'
import { RunStatusBadge } from '@/components/RunStatusBadge'

export function RunsListPage() {
  const { data: runs = [], isLoading } = useQuery({ queryKey: ['runs'], queryFn: listRuns })

  if (isLoading) {
    return (
      <div className="ed-page">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
          Loading runs
        </span>
      </div>
    )
  }

  return (
    <div className="ed-page">
      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888] mb-3">
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

      {runs.length === 0 ? (
        <div className="border border-dashed border-[#d0d0c8] bg-white/40 rounded-[4px] py-16 text-center">
          <p className="font-['DM_Serif_Display'] italic text-[1.4rem] text-[#3a3a36] mb-1">
            No runs yet.
          </p>
          <p className="font-['Outfit'] text-[0.9rem] text-[#888888]">
            Trigger a run from a model's detail page.
          </p>
        </div>
      ) : (
        <ol className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
          {runs.map((run, i) => (
            <li key={run.id}>
              <Link
                to={`/runs/${run.id}`}
                className="flex items-center justify-between gap-5 px-6 py-4 border-b border-[#e5e3d8] last:border-b-0 hover:bg-[#f3f2ec] transition-colors"
              >
                <div className="flex items-baseline gap-4 min-w-0">
                  <span className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#888888] shrink-0">
                    {String(runs.length - i).padStart(3, '0')}
                  </span>
                  <div className="min-w-0">
                    <p className="font-['IBM_Plex_Mono'] text-[0.85rem] text-[#1a1a1a] truncate">
                      {run.workflow_id}
                    </p>
                    <p className="font-['Outfit'] text-[0.78rem] text-[#888888] mt-0.5">
                      {new Date(run.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
                <RunStatusBadge status={run.status} />
              </Link>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
