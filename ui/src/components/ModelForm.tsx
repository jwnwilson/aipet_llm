import { useForm, Controller } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import type { TrainingModel, TrainingModelConfig } from '@/types'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Combobox } from './ui/combobox'

const REMOTE_BACKEND_OPTIONS = ['k8s', 'kaggle', 'ssh', 'colab', 'runpod'] as const

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
  name: z.string().min(1, 'Name is required'),
  description: z.string().default(''),
  base_model: z.string().min(1, 'Base model is required'),
  train_data: z.string().min(1, 'Training data path is required'),
  eval_data: z.string().min(1, 'Eval data path is required'),
  epochs: z.coerce.number().int().min(1),
  patience: z.coerce.number().int().min(1),
  warmup_ratio: z.coerce.number().min(0).max(1),
  remote_backend: z.string().min(1),
  skip_generate: z.boolean().default(false),
})

type FormValues = z.input<typeof schema>

interface ModelFormProps {
  defaultValues?: Partial<TrainingModel>
  onSubmit: (values: TrainingModelConfig) => void
  isSubmitting?: boolean
}

const DEFAULTS: FormValues = {
  name: '',
  description: '',
  base_model: 'HuggingFaceTB/SmolLM2-360M',
  train_data: 'data/train.jsonl',
  eval_data: 'data/eval.jsonl',
  epochs: 5,
  patience: 3,
  warmup_ratio: 0.05,
  remote_backend: 'k8s',
  skip_generate: false,
}

function Field({ label, error, children, hint }: { label: string; error?: string; children: React.ReactNode; hint?: string }) {
  return (
    <div className="flex flex-col gap-2">
      <Label>{label}</Label>
      {children}
      {hint && !error && (
        <p className="font-['Outfit'] text-[0.78rem] text-[#6b6b6b]">{hint}</p>
      )}
      {error && (
        <p className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.12em] text-[#7f1d1d]">
          {error}
        </p>
      )}
    </div>
  )
}

function FormSection({ num, title, children }: { num: string; title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-5">
      <div className="flex items-center gap-3">
        <span className="ed-step-circle ed-step-circle--active text-[0.7rem]">{num}</span>
        <h2 className="font-['DM_Serif_Display'] text-[1.3rem] text-[#1a1a1a]">{title}</h2>
      </div>
      <div className="pl-11 flex flex-col gap-5">
        {children}
      </div>
    </section>
  )
}

export function ModelForm({ defaultValues, onSubmit, isSubmitting = false }: ModelFormProps) {
  const { register, handleSubmit, control, formState: { errors } } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: defaultValues ? { ...DEFAULTS, ...defaultValues } : DEFAULTS,
  })

  return (
    <form
      onSubmit={handleSubmit((values) => onSubmit(values as unknown as TrainingModelConfig))}
      className="flex flex-col gap-10"
    >
      <FormSection num="01" title="Identity">
        <Field label="Name" error={errors.name?.message}>
          <Input {...register('name')} placeholder="my-experiment" />
        </Field>
        <Field label="Description" hint="Optional human-readable summary.">
          <Input {...register('description')} placeholder="A short description of the experiment" />
        </Field>
      </FormSection>

      <hr className="ed-rule m-0" />

      <FormSection num="02" title="Architecture & data">
        <Field label="Base model" error={errors.base_model?.message}>
          <Controller
            name="base_model"
            control={control}
            render={({ field }) => (
              <Combobox
                id="base_model"
                value={field.value ?? ''}
                onChange={field.onChange}
                options={BASE_MODEL_OPTIONS}
                placeholder="HuggingFaceTB/SmolLM2-360M"
                disabled={field.disabled}
              />
            )}
          />
        </Field>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <Field label="Training data path" error={errors.train_data?.message}>
            <Input {...register('train_data')} />
          </Field>
          <Field label="Eval data path" error={errors.eval_data?.message}>
            <Input {...register('eval_data')} />
          </Field>
        </div>
      </FormSection>

      <hr className="ed-rule m-0" />

      <FormSection num="03" title="Hyperparameters">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Field label="Epochs" error={errors.epochs?.message}>
            <Input type="number" {...register('epochs')} />
          </Field>
          <Field label="Patience" error={errors.patience?.message}>
            <Input type="number" {...register('patience')} />
          </Field>
          <Field label="Warmup ratio" error={errors.warmup_ratio?.message}>
            <Input type="number" step="0.01" {...register('warmup_ratio')} />
          </Field>
        </div>
      </FormSection>

      <hr className="ed-rule m-0" />

      <FormSection num="04" title="Execution">
        <Field label="Remote backend" error={errors.remote_backend?.message}>
          <Controller
            name="remote_backend"
            control={control}
            render={({ field }) => (
              <Select value={field.value} onValueChange={field.onChange} disabled={field.disabled}>
                <SelectTrigger onBlur={field.onBlur} ref={field.ref}>
                  <SelectValue placeholder="Select backend" />
                </SelectTrigger>
                <SelectContent>
                  {REMOTE_BACKEND_OPTIONS.map(opt => (
                    <SelectItem key={opt} value={opt}>{opt}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          />
        </Field>
        <label className="flex items-center gap-3 cursor-pointer select-none">
          <input
            type="checkbox"
            id="skip_generate"
            {...register('skip_generate')}
            className="h-4 w-4 accent-[#1a1a1a]"
          />
          <span className="font-['Outfit'] text-[0.92rem] text-[#1a1a1a]">
            Skip dataset generation
          </span>
          <span className="font-['Outfit'] text-[0.82rem] text-[#6b6b6b]">
            — reuse existing data referenced above
          </span>
        </label>
      </FormSection>

      <div className="pt-2 flex items-center gap-3">
        <Button type="submit" disabled={isSubmitting} size="lg">
          {isSubmitting ? 'Saving' : 'Save model'}
        </Button>
        <span className="font-['Outfit'] text-[0.82rem] text-[#6b6b6b]">
          Saved as a draft until you trigger your first run.
        </span>
      </div>
    </form>
  )
}
