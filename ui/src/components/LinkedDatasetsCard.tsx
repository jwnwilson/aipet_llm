import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { listDatasets } from '@/api/datasets'
import { updateModel } from '@/api/models'
import type { Dataset, TrainingModel } from '@/types'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'

// Sentinel value used in the Select to represent "no dataset selected"
const NONE = '__none__'

interface Props {
  model: TrainingModel
}

function DatasetBadge({ dataset }: { dataset: Dataset | undefined }) {
  if (!dataset) return <span className="text-sm text-gray-400 italic">Not linked</span>
  return (
    <div className="flex items-center gap-2">
      <span className="font-medium text-sm text-gray-900">{dataset.name}</span>
      <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
        dataset.dataset_type === 'train'
          ? 'bg-blue-100 text-blue-700'
          : 'bg-green-100 text-green-700'
      }`}>
        {dataset.dataset_type}
      </span>
    </div>
  )
}

export function LinkedDatasetsCard({ model }: Props) {
  const queryClient = useQueryClient()

  const { data: allDatasets = [] } = useQuery({
    queryKey: ['datasets'],
    queryFn: listDatasets,
  })

  const trainDatasets = allDatasets.filter(d => d.dataset_type === 'train')
  const evalDatasets  = allDatasets.filter(d => d.dataset_type === 'eval')

  // Find currently linked datasets by matching storage key
  const linkedTrain = allDatasets.find(d => d.key === model.train_data)
  const linkedEval  = allDatasets.find(d => d.key === model.eval_data)

  // Local selections start from what's currently linked (or NONE)
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
      <CardHeader><CardTitle>Linked datasets</CardTitle></CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label>Training dataset</Label>
          <DatasetBadge dataset={linkedTrain} />
          <Select
            value={selectedTrainId}
            onValueChange={setSelectedTrainId}
            disabled={saveMutation.isPending}
          >
            <SelectTrigger aria-label="Select training dataset">
              <SelectValue placeholder="Select a dataset…" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE}>— None —</SelectItem>
              {trainDatasets.map(ds => (
                <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1.5">
          <Label>Eval dataset</Label>
          <DatasetBadge dataset={linkedEval} />
          <Select
            value={selectedEvalId}
            onValueChange={setSelectedEvalId}
            disabled={saveMutation.isPending}
          >
            <SelectTrigger aria-label="Select eval dataset">
              <SelectValue placeholder="Select a dataset…" />
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
          <p className="text-sm text-red-600">Failed to save. Please try again.</p>
        )}
        {saveMutation.isSuccess && !isDirty && (
          <p className="text-sm text-green-600">Saved.</p>
        )}

        <Button
          onClick={() => saveMutation.mutate()}
          disabled={!isDirty || saveMutation.isPending}
          className="self-start"
        >
          {saveMutation.isPending ? 'Saving…' : 'Save'}
        </Button>
      </CardContent>
    </Card>
  )
}
