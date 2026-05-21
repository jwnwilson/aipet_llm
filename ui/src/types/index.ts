// src/types/index.ts
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
  status: RunStatus
  eval_valid_pct: number | null
  progress: number | null
  progress_detail: string | null
  training_config: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface TriggerRunRequest {
  model_id: string
  epochs?: number | null
  patience?: number | null
  warmup_ratio?: number | null
  skip_generate?: boolean | null
  remote_backend?: string | null
  base_model?: string | null
  num_train_samples?: number | null
  num_eval_samples?: number | null
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
  pet_stats: PetStats
}

export interface InferenceRequest {
  scene: SceneData
}

export interface InferenceResponse {
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
