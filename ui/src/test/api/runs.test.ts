import { describe, it, expect } from 'vitest'
import { listRuns, getRun, isRunActive, isRunCancellable, triggerRun, deleteRun, cancelRun, getRunEvaluation, getRunTemporal, getRunLogs } from '@/api/runs'
import { MODEL_FIXTURE, RUN_FIXTURE, EVAL_DATA_FIXTURE } from '../msw/fixtures'

describe('listRuns', () => {
  it('returns paginated response with RunRecords', async () => {
    const result = await listRuns()
    expect(Array.isArray(result.items)).toBe(true)
    expect(result.items[0].id).toBe(RUN_FIXTURE.id)
  })
})

describe('getRun', () => {
  it('returns run by id', async () => {
    const run = await getRun(RUN_FIXTURE.id)
    expect(run.status).toBe('running')
    expect(run.model_id).toBe(MODEL_FIXTURE.id)
  })

  it('throws on unknown id', async () => {
    await expect(getRun('does-not-exist')).rejects.toThrow()
  })
})

describe('triggerRun', () => {
  it('posts to /api/runs/trigger and returns run_id', async () => {
    const result = await triggerRun({ model_id: MODEL_FIXTURE.id })
    expect(result.run_id).toBe(RUN_FIXTURE.id)
  })
})

describe('isRunActive', () => {
  it('returns true for running status', () => {
    expect(isRunActive({ ...RUN_FIXTURE, status: 'running' })).toBe(true)
  })

  it('returns false for completed status', () => {
    expect(isRunActive({ ...RUN_FIXTURE, status: 'completed' })).toBe(false)
  })

  it('returns false for failed status', () => {
    expect(isRunActive({ ...RUN_FIXTURE, status: 'failed' })).toBe(false)
  })
})

describe('deleteRun', () => {
  it('resolves for an existing run id', async () => {
    await expect(deleteRun(RUN_FIXTURE.id)).resolves.toBeUndefined()
  })

  it('throws for an unknown run id', async () => {
    await expect(deleteRun('does-not-exist')).rejects.toThrow()
  })
})

describe('cancelRun', () => {
  it('resolves for an active run id', async () => {
    await expect(cancelRun(RUN_FIXTURE.id)).resolves.toBeUndefined()
  })

  it('throws for an unknown run id', async () => {
    await expect(cancelRun('does-not-exist')).rejects.toThrow()
  })
})

describe('isRunCancellable', () => {
  it.each(['pending', 'generating', 'training', 'evaluating', 'exporting', 'running'] as const)(
    'returns true for %s status',
    (status) => {
      expect(isRunCancellable({ ...RUN_FIXTURE, status })).toBe(true)
    }
  )

  it.each(['completed', 'failed', 'cancelled'] as const)(
    'returns false for %s status',
    (status) => {
      expect(isRunCancellable({ ...RUN_FIXTURE, status })).toBe(false)
    }
  )
})

describe('getRunEvaluation', () => {
  it('returns EvaluationData for a known run', async () => {
    const result = await getRunEvaluation(EVAL_DATA_FIXTURE.run_id)
    expect(result.run_id).toBe(EVAL_DATA_FIXTURE.run_id)
    expect(result.status).toBe('completed')
    expect(result.eval_valid_pct).toBe(0.97)
  })

  it('throws for an unknown run id', async () => {
    await expect(getRunEvaluation('does-not-exist')).rejects.toThrow()
  })
})

describe('getRunTemporal', () => {
  it('returns TemporalDetails for a known run', async () => {
    const result = await getRunTemporal(RUN_FIXTURE.id)
    expect(result.workflow_id).toBe('training-test-model-abc12345')
    expect(result.temporal_run_id).toBe('temporal-run-id-abc')
    expect(result.status).toBe('RUNNING')
    expect(result.start_time).toBe('2024-01-01T00:00:00.000Z')
    expect(result.close_time).toBeNull()
  })

  it('throws for an unknown run id', async () => {
    await expect(getRunTemporal('does-not-exist')).rejects.toThrow()
  })
})

describe('getRunLogs', () => {
  it('returns RunLogsResponse for a known run', async () => {
    const result = await getRunLogs(RUN_FIXTURE.id)
    expect(result.logs).toBe('epoch 1/3  loss=0.42\n')
    expect(result.source).toBe('local')
  })

  it('throws for an unknown run id', async () => {
    await expect(getRunLogs('does-not-exist')).rejects.toThrow()
  })
})
