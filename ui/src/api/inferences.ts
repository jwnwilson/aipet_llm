import type { InferenceInstance, InferenceRequest, InferenceResponse, PaginatedResponse } from '../types'
import { apiClient } from './client'

export interface InferenceInstanceConfig {
  model_id: string
  pod_name?: string
  pod_namespace?: string
  idle_timeout_minutes?: number
}

export async function listInferences(page = 1, limit = 50, modelId?: string): Promise<PaginatedResponse<InferenceInstance>> {
  const { data } = await apiClient.get<PaginatedResponse<InferenceInstance>>('/api/inferences', {
    params: { page, limit, ...(modelId ? { model_id: modelId } : {}) },
  })
  return data
}

export async function createInference(config: InferenceInstanceConfig): Promise<InferenceInstance> {
  const { data } = await apiClient.post<InferenceInstance>('/api/inferences', config)
  return data
}

export async function getInference(id: string): Promise<InferenceInstance> {
  const { data } = await apiClient.get<InferenceInstance>(`/api/inferences/${id}`)
  return data
}

export async function startInference(id: string): Promise<InferenceInstance> {
  const { data } = await apiClient.post<InferenceInstance>(`/api/inferences/${id}/start`)
  return data
}

export async function stopInference(id: string): Promise<InferenceInstance> {
  const { data } = await apiClient.post<InferenceInstance>(`/api/inferences/${id}/stop`)
  return data
}

export async function deleteInference(id: string): Promise<void> {
  await apiClient.delete(`/api/inferences/${id}`)
}

export async function inferInstance(id: string, request: InferenceRequest): Promise<InferenceResponse> {
  const { data } = await apiClient.post<InferenceResponse>(`/api/inferences/${id}/infer`, request)
  return data
}
