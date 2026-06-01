import type { Dataset, DatasetType, PaginatedResponse } from '@/types'
import { apiClient } from './client'

export interface DatasetUploadResult {
  key: string
}

// ---------------------------------------------------------------------------
// Named dataset CRUD
// ---------------------------------------------------------------------------

export async function listDatasets(page = 1, limit = 50): Promise<PaginatedResponse<Dataset>> {
  const { data } = await apiClient.get<PaginatedResponse<Dataset>>('/api/datasets', { params: { page, limit } })
  return data
}

export async function getDataset(id: string): Promise<Dataset> {
  const { data } = await apiClient.get<Dataset>(`/api/datasets/${id}`)
  return data
}

export async function createDataset(params: {
  name: string
  dataset_type: DatasetType
  description?: string
  file: File
}): Promise<Dataset> {
  const form = new FormData()
  form.append('name', params.name)
  form.append('dataset_type', params.dataset_type)
  form.append('description', params.description ?? '')
  form.append('file', params.file)
  const { data } = await apiClient.post<Dataset>('/api/datasets', form)
  return data
}

export async function deleteDataset(id: string): Promise<void> {
  await apiClient.delete(`/api/datasets/${id}`)
}

// ---------------------------------------------------------------------------
// Legacy fixed-key uploads (backwards compat)
// ---------------------------------------------------------------------------

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
