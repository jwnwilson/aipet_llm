import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Zap } from 'lucide-react'
import { inferInstance } from '@/api/inferences'
import { Button } from '@/components/ui/button'
import type { InferenceRequest, InferenceResponse } from '@/types'

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

interface InstanceInferencePanelProps {
  instanceId: string
}

export function InstanceInferencePanel({ instanceId }: InstanceInferencePanelProps) {
  const [json, setJson] = useState(JSON.stringify(DEFAULT_REQUEST, null, 2))
  const [parseError, setParseError] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: (req: InferenceRequest) => inferInstance(instanceId, req),
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

  return (
    <div className="mt-3 pt-3 border-t border-[#e5e3d8]">
      <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.14em] text-[#888888] mb-2">
        Test inference
      </div>

      <textarea
        className="w-full bg-white px-3 py-2 font-['IBM_Plex_Mono'] text-[0.75rem] text-[#1a1a1a] min-h-36 resize-y border-[1.5px] border-[#d0d0c8] rounded-[3px] focus:outline-none focus:border-[#1a1a1a]"
        value={json}
        onChange={e => { setJson(e.target.value); setParseError(null) }}
        spellCheck={false}
        aria-label="Inference request payload"
      />

      {parseError && (
        <p className="font-['IBM_Plex_Mono'] text-[0.72rem] uppercase tracking-[0.12em] text-[#7f1d1d] mt-1">
          {parseError}
        </p>
      )}

      <div className="mt-2">
        <Button size="sm" onClick={handleRun} disabled={mutation.isPending}>
          <Zap className="h-3 w-3" />
          {mutation.isPending ? 'Running…' : 'Run inference'}
        </Button>
      </div>

      {mutation.isError && (
        <div className="mt-2 border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-3 py-2">
          <p className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#7f1d1d]">
            Inference failed: {String((mutation.error as Error)?.message ?? 'unknown error')}
          </p>
        </div>
      )}

      {mutation.isSuccess && <InferenceResult result={mutation.data} />}
    </div>
  )
}

function InferenceResult({ result }: { result: InferenceResponse }) {
  return (
    <div className="mt-2 bg-[#f6f5ef] border-l-[3px] border-[#1a1a1a] px-4 py-3">
      <div className="font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em] text-[#888888] mb-2">
        Response
      </div>
      <dl className="grid grid-cols-3 gap-x-4 gap-y-1">
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.12em] text-[#888888]">Action</dt>
          <dd className="font-['DM_Serif_Display'] text-[1rem] text-[#1a1a1a]">{result.action}</dd>
        </div>
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.12em] text-[#888888]">Target</dt>
          <dd className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">{result.target_object_id ?? '—'}</dd>
        </div>
        <div>
          <dt className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.12em] text-[#888888]">Confidence</dt>
          <dd className="font-['IBM_Plex_Mono'] text-[0.82rem] text-[#1a1a1a]">
            {result.confidence != null ? `${(result.confidence * 100).toFixed(0)}%` : '—'}
          </dd>
        </div>
      </dl>
    </div>
  )
}
