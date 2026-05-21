import { useRef, useState } from 'react'
import { uploadTrainDataset, uploadEvalDataset } from '@/api/datasets'
import { Button } from './ui/button'
import { Label } from './ui/label'
import { Input } from './ui/input'

export function DatasetUpload() {
  const trainRef = useRef<HTMLInputElement>(null)
  const evalRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState<{ text: string; error: boolean } | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trainFile = trainRef.current?.files?.[0]
    const evalFile = evalRef.current?.files?.[0]

    if (!trainFile && !evalFile) {
      setMessage({ text: 'Select at least one file to upload.', error: true })
      return
    }

    setUploading(true)
    setMessage(null)

    try {
      if (trainFile) await uploadTrainDataset(trainFile)
      if (evalFile) await uploadEvalDataset(evalFile)
      setMessage({ text: 'Uploaded successfully.', error: false })
      if (trainRef.current) trainRef.current.value = ''
      if (evalRef.current) evalRef.current.value = ''
    } catch {
      setMessage({ text: 'Upload failed. Please try again.', error: true })
    } finally {
      setUploading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="train-file">Training dataset</Label>
        <Input
          id="train-file"
          type="file"
          accept=".jsonl"
          ref={trainRef}
          aria-label="Training dataset"
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="eval-file">Eval dataset</Label>
        <Input
          id="eval-file"
          type="file"
          accept=".jsonl"
          ref={evalRef}
          aria-label="Eval dataset"
        />
      </div>
      {message && (
        <p className={`text-sm ${message.error ? 'text-red-600' : 'text-green-600'}`}>
          {message.text}
        </p>
      )}
      <Button type="submit" disabled={uploading} className="self-start">
        {uploading ? 'Uploading…' : 'Upload'}
      </Button>
    </form>
  )
}
