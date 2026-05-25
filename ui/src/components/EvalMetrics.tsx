import { Check, X } from 'lucide-react'
import type { QualityReport } from '@/types'

interface EvalMetricsProps {
  validPct: number
  passed: boolean
  qualityReport?: QualityReport | null
}

export function EvalMetrics({ validPct, passed, qualityReport }: EvalMetricsProps) {
  const pctDisplay = (validPct * 100).toFixed(1)

  return (
    <div className="bg-white border border-[#d0d0c8] rounded-[4px]">
      {/* Headline result */}
      <div className="px-6 py-5 border-b border-[#d0d0c8] flex items-center gap-5">
        <div
          aria-label={passed ? 'passed' : 'failed'}
          role="img"
          className={[
            'flex items-center justify-center h-14 w-14 rounded-full border-[1.5px] shrink-0',
            passed
              ? 'bg-[#e8efe9] border-[#2d6a4f] text-[#2d6a4f]'
              : 'bg-[#f1e2e0] border-[#7f1d1d] text-[#7f1d1d]',
          ].join(' ')}
        >
          {passed ? <Check className="h-6 w-6" /> : <X className="h-6 w-6" />}
        </div>
        <div className="min-w-0">
          <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] mb-1">
            Eval score
          </div>
          <div className="font-['DM_Serif_Display'] text-[2.4rem] leading-none text-[#1a1a1a]">
            {pctDisplay}%
          </div>
          <div
            className={`font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] mt-1 ${
              passed ? 'text-[#2d6a4f]' : 'text-[#7f1d1d]'
            }`}
          >
            {passed ? 'Passed · threshold 95%' : 'Failed · threshold 95%'}
          </div>
        </div>
      </div>

      {qualityReport && (
        <div className="px-6 py-5 flex flex-col gap-6">
          <section>
            <h3 className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] mb-3">
              Per-stat accuracy
            </h3>
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] font-medium text-left pb-2 border-b border-[#1a1a1a]">
                    Stat
                  </th>
                  <th className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] font-medium text-right pb-2 border-b border-[#1a1a1a]">
                    Correct
                  </th>
                  <th className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] font-medium text-right pb-2 border-b border-[#1a1a1a]">
                    Total
                  </th>
                  <th className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] font-medium text-right pb-2 border-b border-[#1a1a1a]">
                    Accuracy
                  </th>
                  <th className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] font-medium text-right pb-2 border-b border-[#1a1a1a]">
                    Pass
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(qualityReport.per_stat_accuracy).map(([stat, result]) => (
                  <tr key={stat} className="border-b border-[#e5e3d8] last:border-b-0">
                    <td className="py-2 font-['Outfit'] text-[0.88rem] text-[#1a1a1a] capitalize">
                      {stat}
                    </td>
                    <td className="py-2 text-right font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">
                      {result.correct}
                    </td>
                    <td className="py-2 text-right font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">
                      {result.total}
                    </td>
                    <td className="py-2 text-right font-['IBM_Plex_Mono'] text-[0.85rem] text-[#1a1a1a]">
                      {(result.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 text-right">
                      {result.passed ? (
                        <Check
                          aria-label="passed"
                          role="img"
                          className="h-3.5 w-3.5 text-[#2d6a4f] inline"
                        />
                      ) : (
                        <X
                          aria-label="failed"
                          role="img"
                          className="h-3.5 w-3.5 text-[#7f1d1d] inline"
                        />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <h3 className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888] mb-3">
              Action distribution
            </h3>
            <div className="flex flex-wrap gap-2">
              {Object.entries(qualityReport.action_distribution).map(([action, count]) => (
                <span
                  key={action}
                  className="inline-flex items-center gap-2 px-2.5 py-1 border border-[#d0d0c8] rounded-[2px] bg-white"
                >
                  <span className="font-['IBM_Plex_Mono'] text-[0.7rem] text-[#1a1a1a] uppercase tracking-[0.08em]">
                    {action}
                  </span>
                  <span className="font-['IBM_Plex_Mono'] text-[0.7rem] text-[#888888]">
                    {count}
                  </span>
                </span>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  )
}
