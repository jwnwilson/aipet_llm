import { CheckCircle, XCircle } from 'lucide-react'
import type { QualityReport } from '@/types'

interface EvalMetricsProps {
  validPct: number
  passed: boolean
  qualityReport?: QualityReport | null
}

export function EvalMetrics({ validPct, passed, qualityReport }: EvalMetricsProps) {
  const pctDisplay = (validPct * 100).toFixed(1)

  return (
    <div className="rounded-md border p-4 space-y-4">
      <div className="flex items-center gap-3">
        {passed
          ? <CheckCircle aria-label="passed" role="img" className="h-5 w-5 text-green-600 shrink-0" />
          : <XCircle aria-label="failed" role="img" className="h-5 w-5 text-red-600 shrink-0" />
        }
        <div>
          <p className="text-sm font-medium">Eval score: {pctDisplay}%</p>
          <p className="text-xs text-gray-500">{passed ? 'Passed (≥95%)' : 'Failed (<95%)'}</p>
        </div>
      </div>

      {qualityReport && (
        <>
          <div>
            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
              Per-stat accuracy
            </h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400">
                  <th className="pb-1 font-normal">Stat</th>
                  <th className="pb-1 font-normal text-right">Correct</th>
                  <th className="pb-1 font-normal text-right">Total</th>
                  <th className="pb-1 font-normal text-right">Accuracy</th>
                  <th className="pb-1 font-normal text-right">Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(qualityReport.per_stat_accuracy).map(([stat, result]) => (
                  <tr key={stat} className="border-t border-gray-100">
                    <td className="py-1 text-gray-700">{stat}</td>
                    <td className="py-1 text-right text-gray-700">{result.correct}</td>
                    <td className="py-1 text-right text-gray-700">{result.total}</td>
                    <td className="py-1 text-right text-gray-700">
                      {(result.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-1 text-right">
                      {result.passed
                        ? <CheckCircle aria-label="passed" role="img" className="h-3.5 w-3.5 text-green-600 inline" />
                        : <XCircle aria-label="failed" role="img" className="h-3.5 w-3.5 text-red-600 inline" />
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
              Action distribution
            </h3>
            <div className="flex flex-wrap gap-2">
              {Object.entries(qualityReport.action_distribution).map(([action, count]) => (
                <span
                  key={action}
                  className="inline-flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 text-xs font-mono text-gray-700"
                >
                  <span>{action}</span>
                  <span className="text-gray-400">{count}</span>
                </span>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
