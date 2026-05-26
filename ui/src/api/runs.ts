import type { EvaluationData, RunLogsResponse, RunRecord, RunStatus, TemporalDetails, TriggerRunRequest } from '@/types'
import { apiClient } from './client'

const ACTIVE_STATUSES = new Set<RunStatus>([
  'pending', 'generating', 'training', 'evaluating', 'exporting', 'running',
])

export async function listRuns(): Promise<RunRecord[]> {
  const { data } = await apiClient.get<RunRecord[]>('/api/runs')
  return data
}

export async function getRun(id: string): Promise<RunRecord> {
  const { data } = await apiClient.get<RunRecord>(`/api/runs/${id}`)
  return data
}

export async function triggerRun(req: TriggerRunRequest): Promise<{ run_id: string }> {
  const { data } = await apiClient.post<{ run_id: string }>('/api/runs/trigger', req)
  return data
}

export async function deleteRun(id: string): Promise<void> {
  await apiClient.delete(`/api/runs/${id}`)
}

export async function cancelRun(id: string): Promise<void> {
  await apiClient.post(`/api/runs/${id}/cancel`)
}

export function isRunActive(run: RunRecord): boolean {
  return run.status === 'running'
}

export function isRunCancellable(run: RunRecord): boolean {
  return ACTIVE_STATUSES.has(run.status)
}

export async function getRunEvaluation(id: string): Promise<EvaluationData> {
  const { data } = await apiClient.get<EvaluationData>(`/api/runs/${id}/evaluation`)
  return data
}

export async function getRunTemporal(id: string): Promise<TemporalDetails> {
  const { data } = await apiClient.get<TemporalDetails>(`/api/runs/${id}/temporal`)
  return data
}

export async function getRunLogs(id: string): Promise<RunLogsResponse> {
  const { data } = await apiClient.get<RunLogsResponse>(`/api/runs/${id}/logs`)
  return data
}
