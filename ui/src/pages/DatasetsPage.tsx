import { useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Trash2, UploadCloud } from 'lucide-react'
import type { Dataset, DatasetType } from '@/types'
import { listDatasets, createDataset, deleteDataset } from '@/api/datasets'
import { Pagination } from '@/components/Pagination'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useMediaQuery } from '@/hooks/useMediaQuery'

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return 'Unexpected error'
}

function DatasetRow({ dataset, onDelete, index }: { dataset: Dataset; onDelete: (id: string) => void; index: number }) {
  return (
    <tr>
      <td className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#888888]">
        {String(index + 1).padStart(2, '0')}
      </td>
      <td className="font-['IBM_Plex_Mono'] text-[0.88rem] text-[#1a1a1a]">{dataset.name}</td>
      <td>
        <span
          className={[
            "inline-flex items-center font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em]",
            'px-2 py-[3px] rounded-[2px] border',
            dataset.dataset_type === 'train'
              ? 'border-[#1a1a1a] bg-[#1a1a1a] text-[#fafaf7]'
              : 'border-[#2d6a4f] bg-[#e8efe9] text-[#2d6a4f]',
          ].join(' ')}
        >
          {dataset.dataset_type}
        </span>
      </td>
      <td className="font-['Outfit'] text-[0.82rem] text-[#3a3a36] max-w-xs truncate">
        {dataset.description || '—'}
      </td>
      <td className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#888888] max-w-xs truncate">
        {dataset.key}
      </td>
      <td className="font-['IBM_Plex_Mono'] text-[0.72rem] text-[#888888]">
        {new Date(dataset.created_at).toLocaleDateString()}
      </td>
      <td className="text-right">
        <button
          onClick={() => onDelete(dataset.id)}
          aria-label={`Delete dataset ${dataset.name}`}
          className="text-[#888888] hover:text-[#7f1d1d] transition-colors p-1.5"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </td>
    </tr>
  )
}

function DatasetMobileCard({
  dataset,
  onDelete,
  index,
}: {
  dataset: Dataset
  onDelete: (id: string) => void
  index: number
}) {
  return (
    <div
      data-testid="dataset-mobile-card"
      className="border-b border-[#e5e3d8] px-4 py-4 bg-white last:border-b-0"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#888888]">
              {String(index + 1).padStart(2, '0')}
            </span>
            <span
              className={[
                "inline-flex items-center font-['IBM_Plex_Mono'] text-[0.62rem] uppercase tracking-[0.14em]",
                'px-2 py-[3px] rounded-[2px] border',
                dataset.dataset_type === 'train'
                  ? 'border-[#1a1a1a] bg-[#1a1a1a] text-[#fafaf7]'
                  : 'border-[#2d6a4f] bg-[#e8efe9] text-[#2d6a4f]',
              ].join(' ')}
            >
              {dataset.dataset_type}
            </span>
          </div>
          <p className="font-['IBM_Plex_Mono'] text-[0.88rem] text-[#1a1a1a]">
            {dataset.name}
          </p>
          {dataset.description && (
            <p className="font-['Outfit'] text-[0.78rem] text-[#888888] mt-0.5 line-clamp-2">
              {dataset.description}
            </p>
          )}
          <p className="font-['IBM_Plex_Mono'] text-[0.68rem] text-[#888888] mt-1 truncate">
            {dataset.key}
          </p>
          <p className="font-['Outfit'] text-[0.72rem] text-[#888888] mt-0.5">
            {new Date(dataset.created_at).toLocaleDateString()}
          </p>
        </div>
        <button
          onClick={() => onDelete(dataset.id)}
          aria-label={`Delete dataset ${dataset.name}`}
          className="text-[#888888] hover:text-[#7f1d1d] transition-colors p-1.5 shrink-0 mt-1"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}

function UploadDropzone({ onSuccess }: { onSuccess: () => void }) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [datasetType, setDatasetType] = useState<DatasetType>('train')
  const [fileName, setFileName] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [message, setMessage] = useState<{ text: string; error: boolean } | null>(null)

  const mutation = useMutation({
    mutationFn: createDataset,
    onSuccess: () => {
      setMessage({ text: 'Dataset uploaded successfully.', error: false })
      setName('')
      setDescription('')
      setDatasetType('train')
      setFileName(null)
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

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file && fileRef.current) {
      const dt = new DataTransfer()
      dt.items.add(file)
      fileRef.current.files = dt.files
      setFileName(file.name)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => fileRef.current?.click()}
        className={[
          'border-[1.5px] border-dashed rounded-[4px] px-6 py-10 text-center cursor-pointer transition-colors',
          dragOver ? 'border-[#1a1a1a] bg-[#f3f2ec]' : 'border-[#1a1a1a] bg-[#fafaf7] hover:bg-[#f6f5ef]',
        ].join(' ')}
      >
        <UploadCloud className="h-8 w-8 text-[#1a1a1a] mx-auto mb-3" />
        <p className="font-['DM_Serif_Display'] text-[1.3rem] text-[#1a1a1a] mb-1">
          {fileName ?? 'Drop your dataset here'}
        </p>
        <p className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#888888]">
          {fileName ? 'Click to replace' : 'Click to browse · accepted format .jsonl'}
        </p>
        <input
          id="dataset-file" aria-label="File"
          type="file"
          accept=".jsonl"
          ref={fileRef}
          onChange={(e) => setFileName(e.target.files?.[0]?.name ?? null)}
          className="hidden"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex flex-col gap-2">
          <Label htmlFor="dataset-name">Name</Label>
          <Input
            id="dataset-name"
            placeholder="e.g. train-v1"
            value={name}
            onChange={e => setName(e.target.value)}
            disabled={mutation.isPending}
          />
        </div>
        <div className="flex flex-col gap-2">
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
      <div className="flex flex-col gap-2">
        <Label htmlFor="dataset-desc">Description (optional)</Label>
        <Input
          id="dataset-desc"
          placeholder="Short description"
          value={description}
          onChange={e => setDescription(e.target.value)}
          disabled={mutation.isPending}
        />
      </div>

      {message && (
        <div
          className={`border-l-[3px] px-3 py-2 ${
            message.error ? 'border-[#7f1d1d] bg-[#f1e2e0]' : 'border-[#2d6a4f] bg-[#e8efe9]'
          }`}
        >
          <p
            className={`font-['IBM_Plex_Mono'] text-[0.76rem] ${
              message.error ? 'text-[#7f1d1d]' : 'text-[#2d6a4f]'
            }`}
          >
            {message.text}
          </p>
        </div>
      )}

      <div>
        <Button type="submit" disabled={mutation.isPending}>
          <UploadCloud className="h-3.5 w-3.5" />
          {mutation.isPending ? 'Uploading' : 'Upload dataset'}
        </Button>
      </div>
    </form>
  )
}

export function DatasetsPage() {
  const queryClient = useQueryClient()
  const [page, setPage] = useState(1)
  const { data: datasetsData, isLoading, error } = useQuery({
    queryKey: ['datasets', page],
    queryFn: () => listDatasets(page),
  })
  const datasets = datasetsData?.items

  const deleteMutation = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['datasets'] }),
  })

  const isMobile = useMediaQuery('(max-width: 767px)')

  function handleDelete(id: string) {
    if (!window.confirm('Delete this dataset? This cannot be undone.')) return
    deleteMutation.mutate(id)
  }

  return (
    <div className="ed-page">
      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888] mb-3">
          Vol. 3 · Corpus
        </div>
        <h1 className="font-['DM_Serif_Display'] text-[2.4rem] leading-[1.05] text-[#1a1a1a] mb-3">
          Datasets
        </h1>
        <p className="font-['Outfit'] text-[1rem] text-[#3a3a36] max-w-2xl leading-relaxed">
          Upload training and evaluation corpora in .jsonl format.
          Datasets are versioned by storage key and can be linked to any number of models.
        </p>
        <hr className="ed-rule mt-7 mb-0" />
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[20rem_1fr] gap-8 items-start">
        {/* Upload column */}
        <aside className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)]">
          <div className="px-6 py-4 border-b border-[#d0d0c8]">
            <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888]">
              Section I
            </div>
            <h2 className="font-['DM_Serif_Display'] text-[1.25rem] text-[#1a1a1a]">
              Upload new
            </h2>
          </div>
          <div className="px-6 py-5">
            <UploadDropzone onSuccess={() => queryClient.invalidateQueries({ queryKey: ['datasets'] })} />
          </div>
        </aside>

        {/* List column */}
        <section className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)]">
          <div className="px-6 py-4 border-b border-[#d0d0c8] flex items-center justify-between">
            <div>
              <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.14em] text-[#888888]">
                Section II
              </div>
              <h2 className="font-['DM_Serif_Display'] text-[1.25rem] text-[#1a1a1a]">
                Catalog
              </h2>
            </div>
            <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#888888]">
              {datasets?.length ?? 0} {datasets?.length === 1 ? 'entry' : 'entries'}
            </span>
          </div>

          <div>
            {isLoading && (
              <div className="px-6 py-8">
                <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
                  Loading
                </span>
              </div>
            )}
            {error && (
              <div className="px-6 py-4 border-l-[3px] border-[#7f1d1d] bg-[#f1e2e0] m-6">
                <p className="font-['IBM_Plex_Mono'] text-[0.78rem] text-[#7f1d1d]">
                  Failed to load datasets.
                </p>
              </div>
            )}
            {datasets && datasets.length === 0 && (
              <div className="px-6 py-12 text-center">
                <p className="font-['DM_Serif_Display'] italic text-[1.2rem] text-[#888888]">
                  No datasets uploaded yet.
                </p>
              </div>
            )}
            {datasets && datasets.length > 0 && (
              <>
              {isMobile ? (
                <div>
                  {datasets.map((ds, i) => (
                    <DatasetMobileCard key={ds.id} dataset={ds} onDelete={handleDelete} index={i} />
                  ))}
                </div>
              ) : (
                <table className="ed-table">
                  <thead>
                    <tr>
                      <th style={{ width: '3rem' }}>№</th>
                      <th>Name</th>
                      <th>Type</th>
                      <th>Description</th>
                      <th>Storage key</th>
                      <th>Created</th>
                      <th style={{ width: '3rem' }}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasets.map((ds, i) => (
                      <DatasetRow key={ds.id} dataset={ds} onDelete={handleDelete} index={i} />
                    ))}
                  </tbody>
                </table>
              )}
              <div className="px-6 pb-4">
                <Pagination page={page} pages={datasetsData?.pages ?? 1} onPageChange={setPage} />
              </div>
              </>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
