import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { createModel, getModel, updateModel } from '@/api/models'
import { ModelForm } from '@/components/ModelForm'
import type { TrainingModelConfig } from '@/types'

export function ModelFormPage() {
  const { id } = useParams<{ id: string }>()
  const isEdit = Boolean(id)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: existing, isLoading } = useQuery({
    queryKey: ['models', id],
    queryFn: () => getModel(id!),
    enabled: isEdit,
  })

  const mutation = useMutation({
    mutationFn: (values: TrainingModelConfig) =>
      isEdit ? updateModel(id!, values) : createModel(values),
    onSuccess: (model) => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
      navigate(`/models/${model.id}`)
    },
  })

  if (isEdit && isLoading) {
    return (
      <div className="ed-page">
        <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#6b6b6b]">
          Loading model
        </span>
      </div>
    )
  }

  return (
    <div className="ed-page max-w-3xl">
      <Link
        to={isEdit ? `/models/${id}` : '/models'}
        className="inline-flex items-center gap-1.5 font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#6b6b6b] hover:text-[#1a1a1a] mb-5"
      >
        <ArrowLeft className="h-3 w-3" />
        {isEdit ? 'Back to model' : 'Back to models'}
      </Link>

      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#6b6b6b] mb-2">
          {isEdit ? 'Edit · Configuration' : 'New · Specification'}
        </div>
        <h1 className="font-['DM_Serif_Display'] text-[2.4rem] leading-[1.05] text-[#1a1a1a] mb-3">
          {isEdit ? 'Edit model' : 'New model'}
        </h1>
        <p className="font-['Outfit'] text-[0.95rem] text-[#3a3a36] max-w-2xl">
          Define the base architecture, training data references, and remote backend.
          All fields can be overridden per-run before launching training.
        </p>
        <hr className="ed-rule mt-7" />
      </header>

      <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] p-8">
        <ModelForm
          defaultValues={existing}
          onSubmit={mutation.mutate}
          isSubmitting={mutation.isPending}
        />
        {mutation.isError && (
          <div className="mt-5 border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] px-4 py-3">
            <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
              Failed to save. Please try again.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
