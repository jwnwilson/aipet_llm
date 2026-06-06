// src/types/index.ts

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  limit: number
  pages: number
}

export interface TrainingModelConfig {
  name: string
  description: string
  base_model: string
  train_data: string
  eval_data: string
  epochs: number
  patience: number
  warmup_ratio: number
  remote_backend: string
  skip_generate: boolean
  gguf_path?: string   // optional — backend defaults to ''
  is_active?: boolean  // optional — backend defaults to false
  backend?: 'local' | 'openrouter'  // optional — backend defaults to 'local'
  backend_model_id?: string  // optional — backend defaults to ''
}

export interface TrainingModel extends TrainingModelConfig {
  id: string
  created_at: string
  updated_at: string
  inference_status?: 'unloaded' | 'ready'
}

export type RunStatus =
  | 'pending'
  | 'generating'
  | 'training'
  | 'evaluating'
  | 'exporting'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface RunRecord {
  id: string
  workflow_id: string
  model_id: string
  name: string | null
  status: RunStatus
  eval_valid_pct: number | null
  progress: number | null
  progress_detail: string | null
  training_config: Record<string, unknown> | null
  train_dataset_id: string | null
  eval_dataset_id: string | null
  created_at: string
  updated_at: string
}

export interface RunLogsResponse {
  logs: string | null
  source: string | null
}

export type DatasetType = 'train' | 'eval'

export interface Dataset {
  id: string
  name: string
  description: string
  dataset_type: DatasetType
  key: string
  created_at: string
  updated_at: string
}

export interface CreateDatasetRequest {
  name: string
  dataset_type: DatasetType
  description?: string
  file: File
}

export interface TriggerRunRequest {
  model_id: string
  name?: string | null
  epochs?: number | null
  patience?: number | null
  warmup_ratio?: number | null
  skip_generate?: boolean | null
  remote_backend?: string | null
  base_model?: string | null
  num_train_samples?: number | null
  num_eval_samples?: number | null
  train_dataset_id?: string | null
  eval_dataset_id?: string | null
}

export interface SceneObject {
  type: 'bowl' | 'bed' | 'toy' | 'player' | 'pet'
  id: string
  distance: number
}

export interface PetStats {
  hunger: number
  tiredness: number
  boredom: number
  social: number
  toilet: number
}

export interface SceneData {
  objects: SceneObject[]
  tick: number
}

export interface InferenceRequest {
  scene: SceneData
  pet_stats: PetStats
}

export interface InferenceResponse {
  stat: string | null
  action: string
  target_object_id: string | null
  confidence: number | null
}

export interface UserContext {
  user_id: string
  email: string | null
  status: 'pending' | 'approved'
}

export interface StatAccuracyResult {
  correct: number
  total: number
  accuracy: number
  passed: boolean
}

export interface CategoryAccuracyResult {
  correct: number
  total: number
  accuracy: number
  passed: boolean
}

export interface QualityReport {
  per_stat_accuracy: Record<string, StatAccuracyResult>
  target_accuracy: CategoryAccuracyResult
  priority_conflict: CategoryAccuracyResult
  fallback_accuracy: CategoryAccuracyResult
  action_distribution: Record<string, number>
  max_action_share: number
  passed: boolean
}

export interface EvaluationData {
  run_id: string
  status: RunStatus
  eval_valid_pct: number | null
  quality_report: QualityReport | null
}

export type InferenceStatus =
  | 'pending'
  | 'initializing'
  | 'available'
  | 'idle'
  | 'shutdown'
  | 'failed'

export interface InferenceInstanceConfig {
  model_id: string
  pod_name?: string
  pod_namespace?: string
  idle_timeout_minutes?: number
}

export interface InferenceInstance {
  id: string
  model_id: string
  run_id: string | null
  pod_name: string
  pod_namespace: string
  idle_timeout_minutes: number
  status: InferenceStatus
  last_used_at: string | null
  created_at: string
  updated_at: string
}

export interface TemporalDetails {
  workflow_id: string
  temporal_run_id: string
  status: string
  start_time: string | null
  close_time: string | null
}

export interface RunLogsResponse {
  logs: string | null
  source: string | null
}
