import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Database } from 'lucide-react'
import { listDatasets } from '@/api/datasets'
import { updateModel } from '@/api/models'
import type { Dataset, TrainingModel } from '@/types'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'

const NONE = '__none__'

interface Props {
  model: TrainingModel
}

function DatasetMeta({ dataset, kind }: { dataset: Dataset | undefined; kind: 'train' | 'eval' }) {
  if (!dataset) {
    return (
      <div className="flex items-center gap-2 py-2 border-l-[3px] border-[#d0d0c8] pl-3 bg-[#f6f5ef]">
        <span className="font-['DM_Serif_Display'] italic text-[0.95rem] text-[#6b6b6b]">
          Not linked
        </span>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-3 py-2 border-l-[3px] border-[#1a1a1a] pl-3 bg-[#f6f5ef]">
      <Database className="h-3.5 w-3.5 text-[#1a1a1a]" />
      <span className="font-['IBM_Plex_Mono'] text-[0.85rem] text-[#1a1a1a] truncate">
        {dataset.name}
      </span>
      <span className="font-['IBM_Plex_Mono'] text-[0.6rem] uppercase tracking-[0.14em] text-[#6b6b6b] ml-auto">
        {kind}
      </span>
    </div>
  )
}

export function LinkedDatasetsCard({ model }: Props) {
  const queryClient = useQueryClient()

  const { data: allDatasetsData } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => listDatasets(),
  })
  const allDatasets = allDatasetsData?.items ?? []

  const trainDatasets = allDatasets.filter(d => d.dataset_type === 'train')
  const evalDatasets  = allDatasets.filter(d => d.dataset_type === 'eval')

  const linkedTrain = allDatasets.find(d => d.key === model.train_data)
  const linkedEval  = allDatasets.find(d => d.key === model.eval_data)

  const [selectedTrainId, setSelectedTrainId] = useState<string>(linkedTrain?.id ?? NONE)
  const [selectedEvalId,  setSelectedEvalId]  = useState<string>(linkedEval?.id  ?? NONE)

  const saveMutation = useMutation({
    mutationFn: () => {
      const trainDs = allDatasets.find(d => d.id === selectedTrainId)
      const evalDs  = allDatasets.find(d => d.id === selectedEvalId)
      return updateModel(model.id, {
        ...model,
        train_data: trainDs?.key ?? model.train_data,
        eval_data:  evalDs?.key  ?? model.eval_data,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models', model.id] })
    },
  })

  const isDirty =
    selectedTrainId !== (linkedTrain?.id ?? NONE) ||
    selectedEvalId  !== (linkedEval?.id  ?? NONE)

  return (
    <Card>
      <CardHeader>
        <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
          Section IV
        </div>
        <CardTitle>Linked datasets</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="flex flex-col gap-2">
          <Label>Training dataset</Label>
          <DatasetMeta dataset={linkedTrain} kind="train" />
          <Select
            value={selectedTrainId}
            onValueChange={setSelectedTrainId}
            disabled={saveMutation.isPending}
          >
            <SelectTrigger aria-label="Select training dataset">
              <SelectValue placeholder="Select a dataset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE}>— None —</SelectItem>
              {trainDatasets.map(ds => (
                <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-2">
          <Label>Eval dataset</Label>
          <DatasetMeta dataset={linkedEval} kind="eval" />
          <Select
            value={selectedEvalId}
            onValueChange={setSelectedEvalId}
            disabled={saveMutation.isPending}
          >
            <SelectTrigger aria-label="Select eval dataset">
              <SelectValue placeholder="Select a dataset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE}>— None —</SelectItem>
              {evalDatasets.map(ds => (
                <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {saveMutation.isError && (
          <p className="font-['IBM_Plex_Mono'] text-[0.72rem] uppercase tracking-[0.12em] text-[#7f1d1d]">
            Failed to save
          </p>
        )}
        {saveMutation.isSuccess && !isDirty && (
          <p className="font-['IBM_Plex_Mono'] text-[0.72rem] uppercase tracking-[0.12em] text-[#2d6a4f]">
            Saved
          </p>
        )}

        <div>
          <Button
            onClick={() => saveMutation.mutate()}
            disabled={!isDirty || saveMutation.isPending}
            variant={isDirty ? 'default' : 'outline'}
          >
            {saveMutation.isPending ? 'Saving' : 'Save links'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
