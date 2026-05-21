import { describe, it, expect } from 'vitest'
import { uploadTrainDataset, uploadEvalDataset } from '@/api/datasets'

const JSONL_CONTENT = '{"prompt":"hello","completion":"world"}\n'

function makeFile(name: string, content: string): File {
  return new File([content], name, { type: 'application/octet-stream' })
}

describe('uploadTrainDataset', () => {
  it('posts to /api/datasets/train and returns key', async () => {
    const result = await uploadTrainDataset(makeFile('train.jsonl', JSONL_CONTENT))
    expect(result.key).toBe('datasets/train.jsonl')
  })
})

describe('uploadEvalDataset', () => {
  it('posts to /api/datasets/eval and returns key', async () => {
    const result = await uploadEvalDataset(makeFile('eval.jsonl', JSONL_CONTENT))
    expect(result.key).toBe('datasets/eval.jsonl')
  })
})
