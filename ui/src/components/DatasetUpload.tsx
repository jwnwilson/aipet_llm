import { useRef, useState } from 'react'
import { UploadCloud } from 'lucide-react'
import { uploadTrainDataset, uploadEvalDataset } from '@/api/datasets'
import { Button } from './ui/button'
import { Label } from './ui/label'
import { Input } from './ui/input'

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return 'Unexpected error'
}

export function DatasetUpload() {
  const trainRef = useRef<HTMLInputElement>(null)
  const evalRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState<{ text: string; error: boolean } | null>(null)

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
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
      if (trainFile) {
        try {
          await uploadTrainDataset(trainFile)
        } catch (err: unknown) {
          throw new Error(`Training upload failed: ${getErrorMessage(err)}`)
        }
      }
      if (evalFile) {
        try {
          await uploadEvalDataset(evalFile)
        } catch (err: unknown) {
          throw new Error(`Eval upload failed: ${getErrorMessage(err)}`)
        }
      }
      setMessage({ text: 'Uploaded successfully.', error: false })
      if (trainRef.current) trainRef.current.value = ''
      if (evalRef.current) evalRef.current.value = ''
    } catch (err: unknown) {
      setMessage({ text: getErrorMessage(err), error: true })
    } finally {
      setUploading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      <div className="border-[1.5px] border-dashed border-[#1a1a1a] rounded-[4px] px-6 py-8 text-center bg-[#fafaf7]">
        <UploadCloud className="h-7 w-7 text-[#1a1a1a] mx-auto mb-3" />
        <p className="font-['DM_Serif_Display'] text-[1.2rem] text-[#1a1a1a] mb-1">
          Drop your dataset here
        </p>
        <p className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#6b6b6b]">
          Accepted format · .jsonl
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <Label htmlFor="train-file">Training dataset</Label>
        <Input
          id="train-file"
          type="file"
          accept=".jsonl"
          ref={trainRef}
          aria-label="Training dataset"
          disabled={uploading}
        />
      </div>
      <div className="flex flex-col gap-2">
        <Label htmlFor="eval-file">Eval dataset</Label>
        <Input
          id="eval-file"
          type="file"
          accept=".jsonl"
          ref={evalRef}
          aria-label="Eval dataset"
          disabled={uploading}
        />
      </div>
      {message && (
        <div
          className={`border-l-[3px] px-3 py-2 ${
            message.error
              ? 'border-[#7f1d1d] bg-[#f1e2e0]'
              : 'border-[#2d6a4f] bg-[#e8efe9]'
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
        <Button type="submit" disabled={uploading}>
          <UploadCloud className="h-3.5 w-3.5" />
          {uploading ? 'Uploading' : 'Upload'}
        </Button>
      </div>
    </form>
  )
}
