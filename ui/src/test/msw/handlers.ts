// apps/llm-ui/src/test/msw/handlers.ts
import { http, HttpResponse } from 'msw'
import type { Dataset, PaginatedResponse, TrainingModel, TrainingModelConfig, TriggerRunRequest, UserContext } from '@/types'

function paginate<T>(items: T[], request: Request): PaginatedResponse<T> {
  const url = new URL(request.url)
  const page = parseInt(url.searchParams.get('page') ?? '1', 10)
  const limit = parseInt(url.searchParams.get('limit') ?? '50', 10)
  const offset = (page - 1) * limit
  const sliced = items.slice(offset, offset + limit)
  const pages = Math.max(1, Math.ceil(items.length / limit))
  return { items: sliced, total: items.length, page, limit, pages }
}
import { MODEL_FIXTURE, RUN_FIXTURE, PENDING_USER_FIXTURE, APPROVED_USER_FIXTURE, EVAL_DATA_FIXTURE, TRAIN_DATASET_FIXTURE, EVAL_DATASET_FIXTURE, TEMPORAL_DETAILS_FIXTURE, RUN_LOGS_FIXTURE } from './fixtures'

const BASE = 'http://localhost:8000'

let models: TrainingModel[] = [MODEL_FIXTURE]
let datasets: Dataset[] = [TRAIN_DATASET_FIXTURE, EVAL_DATASET_FIXTURE]
let pendingUsers: UserContext[] = [PENDING_USER_FIXTURE]
let approvedUsers: UserContext[] = [APPROVED_USER_FIXTURE]

export const handlers = [
  http.get(`${BASE}/api/models`, ({ request }) => HttpResponse.json(paginate(models, request))),

  http.post(`${BASE}/api/models`, async ({ request }) => {
    const config = await request.json() as TrainingModelConfig
    const created: TrainingModel = {
      ...config,
      id: 'new-id',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }
    models = [...models, created]
    return HttpResponse.json(created, { status: 201 })
  }),

  http.get(`${BASE}/api/models/:id`, ({ params }) => {
    const model = models.find(m => m.id === params.id)
    if (!model) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    return HttpResponse.json(model)
  }),

  http.put(`${BASE}/api/models/:id`, async ({ params, request }) => {
    const config = await request.json() as TrainingModelConfig
    const idx = models.findIndex(m => m.id === params.id)
    if (idx === -1) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    const updated = { ...models[idx], ...config, updated_at: new Date().toISOString() }
    models = [...models.slice(0, idx), updated, ...models.slice(idx + 1)]
    return HttpResponse.json(updated)
  }),

  http.delete(`${BASE}/api/models/:id`, ({ params }) => {
    const idx = models.findIndex(m => m.id === params.id)
    if (idx === -1) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    models = models.filter(m => m.id !== params.id)
    return new HttpResponse(null, { status: 204 })
  }),

  http.post(`${BASE}/api/runs/trigger`, async ({ request }) => {
    const body = await request.json() as TriggerRunRequest
    const model = models.find(m => m.id === body.model_id)
    if (!model) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    return HttpResponse.json({ run_id: RUN_FIXTURE.id }, { status: 202 })
  }),

  http.get(`${BASE}/api/runs`, ({ request }) => HttpResponse.json(paginate([RUN_FIXTURE], request))),

  http.get(`${BASE}/api/runs/:id`, ({ params }) => {
    if (params.id === RUN_FIXTURE.id) return HttpResponse.json(RUN_FIXTURE)
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  http.delete(`${BASE}/api/runs/:id`, ({ params }) => {
    if (params.id === RUN_FIXTURE.id) return new HttpResponse(null, { status: 204 })
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  http.post(`${BASE}/api/runs/:id/cancel`, ({ params }) => {
    if (params.id === RUN_FIXTURE.id) return new HttpResponse(null, { status: 204 })
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  http.get(`${BASE}/api/runs/:id/evaluation`, ({ params }) => {
    if (params.id === EVAL_DATA_FIXTURE.run_id) return HttpResponse.json(EVAL_DATA_FIXTURE)
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  http.get(`${BASE}/api/runs/:id/temporal`, ({ params }) => {
    if (params.id === RUN_FIXTURE.id) return HttpResponse.json(TEMPORAL_DETAILS_FIXTURE)
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  http.get(`${BASE}/api/runs/:id/logs`, ({ params }) => {
    if (params.id === RUN_FIXTURE.id) return HttpResponse.json(RUN_LOGS_FIXTURE)
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  // Named dataset CRUD
  http.get(`${BASE}/api/datasets`, ({ request }) => HttpResponse.json(paginate(datasets, request))),

  http.post(`${BASE}/api/datasets`, async () => {
    // Client-side validates name/file before reaching here;
    // return a fixed created dataset so tests can assert success state.
    const created: Dataset = {
      id: `ds-new-${Date.now()}`,
      name: 'uploaded-dataset',
      description: '',
      dataset_type: 'train',
      key: 'datasets/ds-new.jsonl',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }
    datasets = [...datasets, created]
    return HttpResponse.json(created, { status: 201 })
  }),

  http.get(`${BASE}/api/datasets/:id`, ({ params }) => {
    const ds = datasets.find(d => d.id === params.id)
    if (!ds) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    return HttpResponse.json(ds)
  }),

  http.delete(`${BASE}/api/datasets/:id`, ({ params }) => {
    const idx = datasets.findIndex(d => d.id === params.id)
    if (idx === -1) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    datasets = datasets.filter(d => d.id !== params.id)
    return new HttpResponse(null, { status: 204 })
  }),

  // Legacy fixed-key uploads (backwards compat)
  http.post(`${BASE}/api/datasets/train`, async () => {
    return HttpResponse.json({ key: 'datasets/train.jsonl' }, { status: 201 })
  }),

  http.post(`${BASE}/api/datasets/eval`, async () => {
    return HttpResponse.json({ key: 'datasets/eval.jsonl' }, { status: 201 })
  }),

  http.get(`${BASE}/api/admin/users`, ({ request }) => {
    const url = new URL(request.url)
    const status = url.searchParams.get('status') ?? 'approved'
    return HttpResponse.json(status === 'pending' ? pendingUsers : approvedUsers)
  }),

  http.post(`${BASE}/api/admin/users`, async ({ request }) => {
    const body = await request.json() as { user_id: string; email?: string | null }
    const user: UserContext = { user_id: body.user_id, email: body.email ?? null, status: 'approved' }
    approvedUsers = [...approvedUsers, user]
    pendingUsers = pendingUsers.filter(u => u.user_id !== body.user_id)
    return HttpResponse.json({ approved: body.user_id }, { status: 201 })
  }),

  http.delete(`${BASE}/api/admin/users/:userId`, ({ params }) => {
    approvedUsers = approvedUsers.filter(
      u => u.user_id !== decodeURIComponent(params.userId as string)
    )
    return new HttpResponse(null, { status: 204 })
  }),
]

export function resetHandlerState() {
  models = [MODEL_FIXTURE]
  datasets = [TRAIN_DATASET_FIXTURE, EVAL_DATASET_FIXTURE]
  pendingUsers = [PENDING_USER_FIXTURE]
  approvedUsers = [APPROVED_USER_FIXTURE]
}
