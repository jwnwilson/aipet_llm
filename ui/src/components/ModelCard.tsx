import { Play, Pencil } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { TrainingModel } from '@/types'
import { Button } from './ui/button'

interface ModelCardProps {
  model: TrainingModel
  onTrigger: (id: string) => void
  isTriggering?: boolean
}

/**
 * Editorial model card — paper-white surface, ink kicker, serif title,
 * mono data block, ink-button actions.
 */
export function ModelCard({ model, onTrigger, isTriggering = false }: ModelCardProps) {
  return (
    <article className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] flex flex-col transition-shadow hover:shadow-[0_4px_14px_rgba(0,0,0,0.10)]">
      <header className="px-6 pt-5 pb-4 border-b border-[#d0d0c8] flex-1">
        <div className="flex items-center justify-between mb-3">
          <span className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
            Model · <span className="text-[#1a1a1a]">{model.base_model.split('/').pop() ?? model.base_model}</span>
          </span>
          {model.is_active && (
            <span className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.14em] text-[#2d6a4f] border border-[#2d6a4f] px-1.5 py-[1px] rounded-[2px]">
              Active
            </span>
          )}
        </div>
        <h3 className="font-['DM_Serif_Display'] text-[1.5rem] leading-tight text-[#1a1a1a] mb-2">
          <Link to={`/models/${model.id}`} className="hover:underline decoration-[#1a1a1a] underline-offset-4">
            {model.name}
          </Link>
        </h3>
        {model.description && (
          <p className="font-['Outfit'] text-[0.88rem] text-[#3a3a36] leading-snug line-clamp-2">
            {model.description}
          </p>
        )}
      </header>

      <div className="px-6 py-4 grid grid-cols-3 gap-x-4 gap-y-2 border-b border-[#d0d0c8]">
        <div>
          <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.14em] text-[#6b6b6b]">Epochs</div>
          <div className="font-['IBM_Plex_Mono'] text-[0.95rem] text-[#1a1a1a]">{model.epochs}</div>
        </div>
        <div>
          <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.14em] text-[#6b6b6b]">Patience</div>
          <div className="font-['IBM_Plex_Mono'] text-[0.95rem] text-[#1a1a1a]">{model.patience}</div>
        </div>
        <div>
          <div className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.14em] text-[#6b6b6b]">Backend</div>
          <div className="font-['IBM_Plex_Mono'] text-[0.95rem] text-[#1a1a1a] truncate">{model.remote_backend}</div>
        </div>
      </div>

      <footer className="px-6 py-4 flex items-center gap-2">
        <Button
          size="sm"
          onClick={() => onTrigger(model.id)}
          disabled={isTriggering}
          aria-label={`Trigger training run for ${model.name}`}
        >
          <Play className="h-3 w-3" />
          {isTriggering ? 'Starting' : 'Run'}
        </Button>
        <Button size="sm" variant="outline" asChild>
          <Link to={`/models/${model.id}/edit`}>
            <Pencil className="h-3 w-3" />
            Edit
          </Link>
        </Button>
      </footer>
    </article>
  )
}
