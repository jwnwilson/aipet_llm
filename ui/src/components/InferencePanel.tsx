import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { inferModel } from '@/api/models'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { InferenceRequest, InferenceResponse, TrainingModel } from '@/types'
import { Zap } from 'lucide-react'

const DEFAULT_REQUEST: InferenceRequest = {
  scene: {
    objects: [
      { type: 'bowl', id: 'bowl-1', distance: 2.5 },
      { type: 'toy', id: 'toy-1', distance: 4.0 },
    ],
    tick: 1,
  },
  pet_stats: {
    hunger: 0.8,
    tiredness: 0.2,
    boredom: 0.3,
    social: 0.1,
    toilet: 0.0,
  },
}

interface InferencePanelProps {
  model: TrainingModel
}

export function InferencePanel({ model }: InferencePanelProps) {
  const [json, setJson] = useState(JSON.stringify(DEFAULT_REQUEST, null, 2))
  const [parseError, setParseError] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: (req: InferenceRequest) => inferModel(model.id, req),
  })

  function handleRun() {
    setParseError(null)
    let parsed: InferenceRequest
    try {
      parsed = JSON.parse(json) as InferenceRequest
    } catch {
      setParseError('Invalid JSON')
      return
    }
    mutation.mutate(parsed)
  }

  const backendLabel = model.backend === 'openrouter'
    ? `OpenRouter — ${model.backend_model_id || 'unset'}`
    : 'Local GGUF'

  const backendColor = model.backend === 'openrouter'
    ? 'bg-purple-100 text-purple-800'
    : 'bg-green-100 text-green-800'

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-4 w-4" />
          Run inference
        </CardTitle>
        <span className={`text-xs font-medium px-2 py-1 rounded-full ${backendColor}`}>
          {backendLabel}
        </span>
        {model.backend !== 'openrouter' && model.inference_status !== 'ready' && (
          <p className="text-xs text-amber-600 mt-1">
            ⚠ Model is not loaded — activate the model before running inference.
          </p>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <textarea
          className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs min-h-48 resize-y focus:outline-none focus:ring-2 focus:ring-ring"
          value={json}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setJson(e.target.value)}
          spellCheck={false}
        />
        {parseError && <p className="text-xs text-red-600">{parseError}</p>}
        <Button onClick={handleRun} disabled={mutation.isPending} className="self-start">
          {mutation.isPending ? 'Running…' : 'Run'}
        </Button>

        {mutation.isError && (
          <p className="text-sm text-red-600">
            {String((mutation.error as Error)?.message ?? 'Inference failed')}
          </p>
        )}

        {mutation.isSuccess && <InferenceResult result={mutation.data} />}
      </CardContent>
    </Card>
  )
}

function InferenceResult({ result }: { result: InferenceResponse }) {
  return (
    <div className="rounded-md border bg-gray-50 p-4 text-sm">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2">
        <dt className="text-gray-500">Action</dt>
        <dd className="font-semibold text-gray-900">{result.action}</dd>

        <dt className="text-gray-500">Target</dt>
        <dd className="font-mono text-gray-900">{result.target_object_id ?? '—'}</dd>

        <dt className="text-gray-500">Confidence</dt>
        <dd className="text-gray-900">
          {result.confidence != null ? `${(result.confidence * 100).toFixed(0)}%` : '—'}
        </dd>
      </dl>
    </div>
  )
}
