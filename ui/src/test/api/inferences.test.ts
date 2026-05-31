import { describe, it, expect } from 'vitest'
import { inferInstance } from '@/api/inferences'
import { server } from '@/test/msw/server'
import { http, HttpResponse } from 'msw'

const BASE = 'http://localhost:8000'

describe('inferInstance', () => {
  it('posts to the correct endpoint and returns the response', async () => {
    const mockResponse = { action: 'EAT', stat: null, target_object_id: 'bowl-1', confidence: 0.9 }
    server.use(
      http.post(`${BASE}/api/inferences/inst-1/infer`, () => HttpResponse.json(mockResponse)),
    )
    const req = {
      scene: { objects: [{ type: 'bowl' as const, id: 'bowl-1', distance: 2.0 }], tick: 1 },
      pet_stats: { hunger: 0.8, tiredness: 0.1, boredom: 0.2, social: 0.0, toilet: 0.0 },
    }
    const result = await inferInstance('inst-1', req)
    expect(result.action).toBe('EAT')
    expect(result.target_object_id).toBe('bowl-1')
  })
})
