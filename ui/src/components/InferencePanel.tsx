import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Zap, AlertCircle } from 'lucide-react'
import { inferModel } from '@/api/models'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { InferenceRequest, InferenceResponse, TrainingModel } from '@/types'

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
    ? `OpenRouter · ${model.backend_model_id || 'unset'}`
    : 'Local GGUF'

  const needsActivation = model.backend !== 'openrouter' && model.inference_status !== 'ready'

  return (
    <Card>
      <CardHeader>
        <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
          Section II
        </div>
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-[18px] w-[18px] text-[#1a1a1a]" />
            Inference
          </CardTitle>
          <span className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] px-2 py-[3px] border border-[#d0d0c8] rounded-[2px] text-[#3a3a36]">
            {backendLabel}
          </span>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {needsActivation && (
          <div className="flex items-start gap-2 border-l-[3px] border-[#92400e] bg-[#f4ecd8] px-3 py-2">
            <AlertCircle className="h-4 w-4 text-[#92400e] mt-0.5 shrink-0" />
            <p className="font-['Outfit'] text-[0.82rem] text-[#92400e] leading-snug">
              Model is not loaded. Activate the model before running inference.
            </p>
          </div>
        )}

        <div>
          <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#6b6b6b] mb-1.5">
            Request payload
          </div>
          <textarea
            className="w-full bg-white px-3 py-2 font-['IBM_Plex_Mono'] text-[0.78rem] text-[#1a1a1a] min-h-48 resize-y border-[1.5px] border-[#d0d0c8] rounded-[3px] focus:outline-none focus:border-[#1a1a1a]"
            value={json}
            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setJson(e.target.value)}
            spellCheck={false}
          />
        </div>

        {parseError && (
          <p className="font-['IBM_Plex_Mono'] text-[0.72rem] uppercase tracking-[0.12em] text-[#7f1d1d]">
            {parseError}
          </p>
        )}

        <div>
          <Button onClick={handleRun} disabled={mutation.isPending}>
            <Zap className="h-3.5 w-3.5" />
            {mutation.isPending ? 'Inferring' : 'Run inference'}
          </Button>
        </div>

        {mutation.isError && (
          <div className="border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-3 py-2">
            <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
              {String((mutation.error as Error)?.message ?? 'Inference failed')}
            </p>
          </div>
        )}

        {mutation.isSuccess && <InferenceResult result={mutation.data} />}
      </CardContent>
    </Card>
  )
}

function InferenceResult({ result }: { result: InferenceResponse }) {
  return (
    <div className="bg-[#f6f5ef] border-l-[3px] border-[#1a1a1a] px-4 py-3">
      <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#6b6b6b] mb-2">
        Response
      </div>
      <dl className="grid grid-cols-3 gap-x-6 gap-y-1">
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
            Action
          </dt>
          <dd className="font-['DM_Serif_Display'] text-[1.1rem] text-[#1a1a1a]">{result.action}</dd>
        </div>
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
            Target
          </dt>
          <dd className="font-['IBM_Plex_Mono'] text-[0.9rem] text-[#1a1a1a]">
            {result.target_object_id ?? '—'}
          </dd>
        </div>
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
            Confidence
          </dt>
          <dd className="font-['IBM_Plex_Mono'] text-[0.9rem] text-[#1a1a1a]">
            {result.confidence != null ? `${(result.confidence * 100).toFixed(0)}%` : '—'}
          </dd>
        </div>
      </dl>
    </div>
  )
}
