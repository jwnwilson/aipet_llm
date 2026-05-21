import { useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Trash2, Upload } from 'lucide-react'
import type { Dataset, DatasetType } from '@/types'
import { listDatasets, createDataset, deleteDataset } from '@/api/datasets'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return 'Unexpected error'
}

function DatasetRow({ dataset, onDelete }: { dataset: Dataset; onDelete: (id: string) => void }) {
  return (
    <tr className="border-b last:border-0">
      <td className="py-3 pr-4 font-medium">{dataset.name}</td>
      <td className="py-3 pr-4">
        <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
          dataset.dataset_type === 'train'
            ? 'bg-blue-100 text-blue-700'
            : 'bg-green-100 text-green-700'
        }`}>
          {dataset.dataset_type}
        </span>
      </td>
      <td className="py-3 pr-4 text-sm text-gray-500 max-w-xs truncate">{dataset.description || '—'}</td>
      <td className="py-3 pr-4 text-sm text-gray-400 font-mono truncate max-w-xs">{dataset.key}</td>
      <td className="py-3 text-sm text-gray-400">
        {new Date(dataset.created_at).toLocaleDateString()}
      </td>
      <td className="py-3 pl-4">
        <button
          onClick={() => onDelete(dataset.id)}
          aria-label={`Delete dataset ${dataset.name}`}
          className="text-gray-400 hover:text-red-500 transition-colors"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </td>
    </tr>
  )
}

function UploadForm({ onSuccess }: { onSuccess: () => void }) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [datasetType, setDatasetType] = useState<DatasetType>('train')
  const [message, setMessage] = useState<{ text: string; error: boolean } | null>(null)

  const mutation = useMutation({
    mutationFn: createDataset,
    onSuccess: () => {
      setMessage({ text: 'Dataset uploaded successfully.', error: false })
      setName('')
      setDescription('')
      setDatasetType('train')
      if (fileRef.current) fileRef.current.value = ''
      onSuccess()
    },
    onError: (err: unknown) => {
      setMessage({ text: getErrorMessage(err), error: true })
    },
  })

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const file = fileRef.current?.files?.[0]
    if (!file) {
      setMessage({ text: 'Please select a file.', error: true })
      return
    }
    if (!name.trim()) {
      setMessage({ text: 'Please enter a name.', error: true })
      return
    }
    setMessage(null)
    mutation.mutate({ name: name.trim(), dataset_type: datasetType, description, file })
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="dataset-name">Name</Label>
          <Input
            id="dataset-name"
            placeholder="e.g. train-v1"
            value={name}
            onChange={e => setName(e.target.value)}
            disabled={mutation.isPending}
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label>Type</Label>
          <Select
            value={datasetType}
            onValueChange={(v) => setDatasetType(v as DatasetType)}
            disabled={mutation.isPending}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="train">Train</SelectItem>
              <SelectItem value="eval">Eval</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="dataset-desc">Description (optional)</Label>
        <Input
          id="dataset-desc"
          placeholder="Short description…"
          value={description}
          onChange={e => setDescription(e.target.value)}
          disabled={mutation.isPending}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="dataset-file">File (.jsonl)</Label>
        <Input
          id="dataset-file"
          type="file"
          accept=".jsonl"
          ref={fileRef}
          disabled={mutation.isPending}
        />
      </div>
      {message && (
        <p className={`text-sm ${message.error ? 'text-red-600' : 'text-green-600'}`}>
          {message.text}
        </p>
      )}
      <Button type="submit" disabled={mutation.isPending} className="self-start flex items-center gap-2">
        <Upload className="h-4 w-4" />
        {mutation.isPending ? 'Uploading…' : 'Upload dataset'}
      </Button>
    </form>
  )
}

export function DatasetsPage() {
  const queryClient = useQueryClient()
  const { data: datasets, isLoading, error } = useQuery({
    queryKey: ['datasets'],
    queryFn: listDatasets,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['datasets'] }),
  })

  function handleDelete(id: string) {
    if (!window.confirm('Delete this dataset? This cannot be undone.')) return
    deleteMutation.mutate(id)
  }

  return (
    <div className="max-w-5xl mx-auto px-8 py-8 flex flex-col gap-8">
      <h1 className="text-2xl font-semibold">Datasets</h1>

      {/* Upload form */}
      <section className="bg-white rounded-lg border p-6">
        <h2 className="text-base font-medium mb-4">Upload new dataset</h2>
        <UploadForm onSuccess={() => queryClient.invalidateQueries({ queryKey: ['datasets'] })} />
      </section>

      {/* Dataset list */}
      <section className="bg-white rounded-lg border p-6">
        <h2 className="text-base font-medium mb-4">Your datasets</h2>
        {isLoading && <p className="text-sm text-gray-500">Loading…</p>}
        {error && <p className="text-sm text-red-600">Failed to load datasets.</p>}
        {datasets && datasets.length === 0 && (
          <p className="text-sm text-gray-500">No datasets uploaded yet.</p>
        )}
        {datasets && datasets.length > 0 && (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500 text-xs uppercase tracking-wider">
                <th className="pb-2 pr-4">Name</th>
                <th className="pb-2 pr-4">Type</th>
                <th className="pb-2 pr-4">Description</th>
                <th className="pb-2 pr-4">Storage key</th>
                <th className="pb-2">Created</th>
                <th className="pb-2 pl-4" />
              </tr>
            </thead>
            <tbody>
              {datasets.map(ds => (
                <DatasetRow key={ds.id} dataset={ds} onDelete={handleDelete} />
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
