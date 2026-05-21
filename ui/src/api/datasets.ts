import { apiClient } from './client'

export interface DatasetUploadResult {
  key: string
}

export async function uploadTrainDataset(file: File): Promise<DatasetUploadResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await apiClient.post<DatasetUploadResult>('/api/datasets/train', form)
  return data
}

export async function uploadEvalDataset(file: File): Promise<DatasetUploadResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await apiClient.post<DatasetUploadResult>('/api/datasets/eval', form)
  return data
}
