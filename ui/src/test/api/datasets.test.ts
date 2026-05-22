/**
 * Tests for the datasets API module.
 *
 * We spy on apiClient.post instead of going through MSW end-to-end for
 * FormData requests, because JSDOM's FormData is incompatible with undici's
 * Request constructor that MSW uses internally — passing it raw causes the
 * request to hang. The spy approach tests the same contract: that the
 * function builds the correct FormData and calls apiClient with it.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import type { Dataset } from '@/types'
import { uploadTrainDataset, uploadEvalDataset, createDataset } from '@/api/datasets'
import { apiClient } from '@/api/client'

const JSONL_CONTENT = '{"prompt":"hello","completion":"world"}\n'

function makeFile(name: string, content: string): File {
  return new File([content], name, { type: 'application/octet-stream' })
}

afterEach(() => {
  vi.restoreAllMocks()
})

// ---------------------------------------------------------------------------
// apiClient default headers
// ---------------------------------------------------------------------------

describe('apiClient configuration', () => {
  it('does not set Content-Type: application/json as a default header', () => {
    // If Content-Type: application/json is in the defaults, axios will
    // JSON-serialize any FormData body instead of sending multipart/form-data.
    // This causes FastAPI to receive null for all Form(...) / File(...) fields
    // and return 422.
    const headers = apiClient.defaults.headers as Record<string, unknown>
    const commonType = (headers?.common as Record<string, unknown>)?.['Content-Type']
    const rootType = headers?.['Content-Type']
    const postType = (headers?.post as Record<string, unknown>)?.['Content-Type']
    expect(commonType).not.toBe('application/json')
    expect(rootType).not.toBe('application/json')
    expect(postType).not.toBe('application/json')
  })
})

// ---------------------------------------------------------------------------
// uploadTrainDataset
// ---------------------------------------------------------------------------

describe('uploadTrainDataset', () => {
  it('posts FormData to /api/datasets/train and returns the key', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({
      data: { key: 'datasets/train.jsonl' },
    })

    const result = await uploadTrainDataset(makeFile('train.jsonl', JSONL_CONTENT))

    expect(apiClient.post).toHaveBeenCalledWith(
      '/api/datasets/train',
      expect.any(FormData),
    )
    expect(result.key).toBe('datasets/train.jsonl')
  })

  it('includes the file in the FormData', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({
      data: { key: 'datasets/train.jsonl' },
    })

    const file = makeFile('train.jsonl', JSONL_CONTENT)
    await uploadTrainDataset(file)

    const fd = (apiClient.post as ReturnType<typeof vi.spyOn>).mock.calls[0][1] as FormData
    expect(fd.get('file')).toBe(file)
  })
})

// ---------------------------------------------------------------------------
// uploadEvalDataset
// ---------------------------------------------------------------------------

describe('uploadEvalDataset', () => {
  it('posts FormData to /api/datasets/eval and returns the key', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({
      data: { key: 'datasets/eval.jsonl' },
    })

    const result = await uploadEvalDataset(makeFile('eval.jsonl', JSONL_CONTENT))

    expect(apiClient.post).toHaveBeenCalledWith(
      '/api/datasets/eval',
      expect.any(FormData),
    )
    expect(result.key).toBe('datasets/eval.jsonl')
  })
})

// ---------------------------------------------------------------------------
// createDataset
// ---------------------------------------------------------------------------

describe('createDataset', () => {
  function makeCreated(overrides: Partial<Dataset> = {}): Dataset {
    return {
      id: 'ds-new-1',
      name: 'test-ds',
      description: '',
      dataset_type: 'train',
      key: 'datasets/ds-new-1.jsonl',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      ...overrides,
    }
  }

  it('posts FormData (not a JSON string) to /api/datasets', async () => {
    // Root cause of the 422: axios JSON-serialises FormData when
    // Content-Type: application/json is set as a default header, so FastAPI
    // receives a JSON body instead of multipart/form-data and sees null for
    // all Form/File fields.  The fix is to remove that default so axios sends
    // FormData as-is and the browser XHR sets multipart/form-data with the
    // correct boundary automatically.
    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({ data: makeCreated() })

    await createDataset({
      name: 'test-ds',
      dataset_type: 'train',
      description: 'a description',
      file: makeFile('data.jsonl', JSONL_CONTENT),
    })

    const [url, body] = (apiClient.post as ReturnType<typeof vi.spyOn>).mock.calls[0]
    expect(url).toBe('/api/datasets')
    // Body MUST be FormData — if it's a string the server receives JSON and returns 422
    expect(body).toBeInstanceOf(FormData)
  })

  it('includes all required fields in the FormData', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({ data: makeCreated() })

    const file = makeFile('data.jsonl', JSONL_CONTENT)
    await createDataset({
      name: 'my-dataset',
      dataset_type: 'eval',
      description: 'test description',
      file,
    })

    const fd = (apiClient.post as ReturnType<typeof vi.spyOn>).mock.calls[0][1] as FormData
    expect(fd.get('name')).toBe('my-dataset')
    expect(fd.get('dataset_type')).toBe('eval')
    expect(fd.get('description')).toBe('test description')
    expect(fd.get('file')).toBe(file)
  })

  it('defaults description to an empty string when omitted', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({ data: makeCreated() })

    await createDataset({
      name: 'no-desc',
      dataset_type: 'train',
      file: makeFile('data.jsonl', JSONL_CONTENT),
    })

    const fd = (apiClient.post as ReturnType<typeof vi.spyOn>).mock.calls[0][1] as FormData
    expect(fd.get('description')).toBe('')
  })

  it('returns the created dataset record from the response', async () => {
    const expected = makeCreated({ name: 'my-ds', dataset_type: 'eval' })
    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({ data: expected })

    const result = await createDataset({
      name: 'my-ds',
      dataset_type: 'eval',
      file: makeFile('data.jsonl', JSONL_CONTENT),
    })

    expect(result).toEqual(expected)
  })
})
