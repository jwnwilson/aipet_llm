import { useForm, Controller } from 'react-hook-form'
import type { Resolver } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { X, Play } from 'lucide-react'
import type { TrainingModel } from '@/types'
import { triggerRun } from '@/api/runs'
import { listDatasets } from '@/api/datasets'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Combobox } from './ui/combobox'

const REMOTE_BACKEND_OPTIONS = ['local', 'kaggle', 'ssh', 'colab', 'runpod', 'vastai'] as const

const BASE_MODEL_OPTIONS = [
  'HuggingFaceTB/SmolLM2-360M',
  'HuggingFaceTB/SmolLM2-1.7B',
  'Qwen/Qwen2.5-0.5B',
  'Qwen/Qwen2.5-1.5B',
  'microsoft/phi-2',
  'google/gemma-2-2b',
  'meta-llama/Llama-3.2-1B',
  'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
]

const schema = z.object({
  epochs:             z.coerce.number().int().positive().nullable(),
  patience:           z.coerce.number().int().positive().nullable(),
  warmup_ratio:       z.coerce.number().min(0).max(1).nullable(),
  remote_backend:     z.string().nullable(),
  base_model:         z.string().nullable(),
  skip_generate:      z.boolean(),
  num_train_samples:  z.coerce.number().int().positive().nullable(),
  num_eval_samples:   z.coerce.number().int().positive().nullable(),
  train_dataset_id:   z.string().nullable(),
  eval_dataset_id:    z.string().nullable(),
})

type FormValues = z.infer<typeof schema>

interface RunModalProps {
  model: TrainingModel
  onClose: () => void
}

export function RunModal({ model, onClose }: RunModalProps) {
  const queryClient = useQueryClient()

  const { register, handleSubmit, control, watch } = useForm<FormValues>({
    resolver: zodResolver(schema) as Resolver<FormValues>,
    defaultValues: {
      epochs:            model.epochs,
      patience:          model.patience,
      warmup_ratio:      model.warmup_ratio,
      remote_backend:    model.remote_backend,
      base_model:        model.base_model,
      skip_generate:     model.skip_generate,
      num_train_samples: null,
      num_eval_samples:  null,
      train_dataset_id:  null,
      eval_dataset_id:   null,
    },
  })

  const skipGenerate = watch('skip_generate')

  const { data: allDatasets = [] } = useQuery({
    queryKey: ['datasets'],
    queryFn: listDatasets,
  })
  const trainDatasets = allDatasets.filter(d => d.dataset_type === 'train')
  const evalDatasets  = allDatasets.filter(d => d.dataset_type === 'eval')

  const mutation = useMutation({
    mutationFn: triggerRun,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      onClose()
    },
  })

  function onSubmit(values: FormValues) {
    mutation.mutate({
      model_id: model.id,
      ...(values.epochs             != null && { epochs:            values.epochs }),
      ...(values.patience           != null && { patience:          values.patience }),
      ...(values.warmup_ratio       != null && { warmup_ratio:      values.warmup_ratio }),
      ...(values.remote_backend     != null && { remote_backend:    values.remote_backend }),
      ...(values.base_model         != null && { base_model:        values.base_model }),
      ...(!values.skip_generate && values.num_train_samples != null && { num_train_samples: values.num_train_samples }),
      ...(!values.skip_generate && values.num_eval_samples  != null && { num_eval_samples:  values.num_eval_samples }),
      skip_generate: values.skip_generate,
      ...(values.train_dataset_id   != null && { train_dataset_id:  values.train_dataset_id }),
      ...(values.eval_dataset_id    != null && { eval_dataset_id:   values.eval_dataset_id }),
    })
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[#1a1a1a]/35 backdrop-blur-[2px] p-4"
      onClick={onClose}
    >
      <div
        className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_10px_40px_rgba(0,0,0,0.20)] w-full max-w-lg max-h-[92vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="run-modal-title"
      >
        <header className="px-6 py-5 border-b border-[#d0d0c8] flex items-start justify-between">
          <div>
            <div className="font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.18em] text-[#888888] mb-1">
              New training run
            </div>
            <h2 id="run-modal-title" className="font-['DM_Serif_Display'] text-[1.5rem] leading-tight text-[#1a1a1a]">
              Trigger run — {model.name}
            </h2>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-[#888888] hover:text-[#1a1a1a] transition-colors p-1 -m-1"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="px-6 py-5">
          <p className="font-['Outfit'] text-[0.85rem] text-[#3a3a36] mb-5 leading-relaxed">
            Override configuration values for this run only. Empty fields use model defaults.
          </p>

          <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="epochs">Epochs</Label>
                <Input id="epochs" type="number" {...register('epochs')} />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="patience">Patience</Label>
                <Input id="patience" type="number" {...register('patience')} />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="warmup_ratio">Warmup ratio</Label>
                <Input id="warmup_ratio" type="number" step="0.01" {...register('warmup_ratio')} />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Remote backend</Label>
                <Controller
                  name="remote_backend"
                  control={control}
                  render={({ field }) => (
                    <Select value={field.value ?? ''} onValueChange={field.onChange}>
                      <SelectTrigger onBlur={field.onBlur} ref={field.ref}>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {REMOTE_BACKEND_OPTIONS.map(opt => (
                          <SelectItem key={opt} value={opt}>{opt}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="num_train_samples">Train samples</Label>
                <Input
                  id="num_train_samples"
                  type="number"
                  {...register('num_train_samples')}
                  disabled={skipGenerate}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="num_eval_samples">Eval samples</Label>
                <Input
                  id="num_eval_samples"
                  type="number"
                  {...register('num_eval_samples')}
                  disabled={skipGenerate}
                />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label>Base model</Label>
              <Controller
                name="base_model"
                control={control}
                render={({ field }) => (
                  <Combobox
                    value={field.value ?? ''}
                    onChange={field.onChange}
                    options={BASE_MODEL_OPTIONS}
                  />
                )}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1.5">
                <Label>Train dataset</Label>
                <Controller
                  name="train_dataset_id"
                  control={control}
                  render={({ field }) => (
                    <Select value={field.value ?? ''} onValueChange={v => field.onChange(v || null)}>
                      <SelectTrigger onBlur={field.onBlur} ref={field.ref} aria-label="Train dataset">
                        <SelectValue placeholder="Model default" />
                      </SelectTrigger>
                      <SelectContent>
                        {trainDatasets.map(ds => (
                          <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Eval dataset</Label>
                <Controller
                  name="eval_dataset_id"
                  control={control}
                  render={({ field }) => (
                    <Select value={field.value ?? ''} onValueChange={v => field.onChange(v || null)}>
                      <SelectTrigger onBlur={field.onBlur} ref={field.ref} aria-label="Eval dataset">
                        <SelectValue placeholder="Model default" />
                      </SelectTrigger>
                      <SelectContent>
                        {evalDatasets.map(ds => (
                          <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
            </div>

            <label className="flex items-center gap-2 cursor-pointer select-none mt-1">
              <input
                type="checkbox"
                id="modal_skip_generate"
                {...register('skip_generate')}
                className="h-4 w-4 accent-[#1a1a1a]"
              />
              <span className="font-['Outfit'] text-[0.88rem] text-[#1a1a1a]">
                Skip dataset generation
              </span>
            </label>

            {mutation.isError && (
              <div className="border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-3 py-2">
                <p className="font-['IBM_Plex_Mono'] text-[0.76rem] text-[#7f1d1d]">
                  Failed to start run. Please try again.
                </p>
              </div>
            )}

            <div className="flex justify-end gap-2 mt-3 pt-3 border-t border-[#d0d0c8]">
              <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
              <Button type="submit" disabled={mutation.isPending}>
                <Play className="h-3.5 w-3.5" />
                {mutation.isPending ? 'Starting' : 'Start run'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
