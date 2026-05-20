// apps/llm-ui/src/test/msw/fixtures.ts
import type { TrainingModel, RunRecord, UserContext, QualityReport, EvaluationData } from '@/types'

export const MODEL_FIXTURE: TrainingModel = {
  id: 'test-id-1',
  name: 'test-model',
  description: 'A test model',
  base_model: 'HuggingFaceTB/SmolLM2-360M',
  train_data: 'data/train.jsonl',
  eval_data: 'data/eval.jsonl',
  epochs: 5,
  patience: 3,
  warmup_ratio: 0.05,
  remote_backend: 'local',
  skip_generate: false,
  gguf_path: '',
  is_active: false,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
}

export const RUN_FIXTURE: RunRecord = {
  id: 'run-uuid',
  workflow_id: 'training-test-model-abc12345',
  model_id: 'test-id-1',
  status: 'running',
  eval_valid_pct: null,
  progress: null,
  progress_detail: null,
  training_config: null,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
}

export const PENDING_USER_FIXTURE: UserContext = {
  user_id: 'auth0|pending-user',
  email: 'pending@example.com',
  status: 'pending',
}

export const APPROVED_USER_FIXTURE: UserContext = {
  user_id: 'auth0|approved-user',
  email: 'approved@example.com',
  status: 'approved',
}

export const QUALITY_REPORT_FIXTURE: QualityReport = {
  per_stat_accuracy: {
    hunger:    { correct: 38, total: 40, accuracy: 0.95,  passed: true  },
    boredom:   { correct: 37, total: 40, accuracy: 0.925, passed: true  },
    social:    { correct: 39, total: 40, accuracy: 0.975, passed: true  },
    tiredness: { correct: 36, total: 40, accuracy: 0.9,   passed: false },
    toilet:    { correct: 38, total: 40, accuracy: 0.95,  passed: true  },
  },
  target_accuracy:   { correct: 18, total: 20, accuracy: 0.9,  passed: true },
  priority_conflict: { correct: 16, total: 20, accuracy: 0.8,  passed: true },
  fallback_accuracy: { correct: 19, total: 20, accuracy: 0.95, passed: true },
  action_distribution: { EAT: 50, SLEEP: 40, PLAY: 10 },
  max_action_share: 0.5,
  passed: true,
}

export const EVAL_DATA_FIXTURE: EvaluationData = {
  run_id: 'run-uuid',
  status: 'completed',
  eval_valid_pct: 0.97,
  quality_report: QUALITY_REPORT_FIXTURE,
}
