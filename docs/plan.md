# Plan

> Completed work → [complete.md](complete.md)

---

## EPIC-10: LLM API — API Keys & Rate Limiting

> Per-user API keys and rate limiting for the inference endpoints.

### TASK-10.1 — Issue API keys per Auth0 user
Add `POST /api/keys` (create key) and `GET /api/keys` (list user's keys) endpoints. Store hashed keys in the DB linked to the Auth0 user ID. Keys are presented once on creation.

**Outputs:** `src/interactors/api/routes/keys.py`, `src/adapters/database/key_store.py`, DB migration

### TASK-10.2 — Accept API key as `Authorization: Bearer` on inference endpoints
Allow inference endpoints to authenticate via either a JWT (Auth0) or a raw API key. Add key lookup to the `require_auth` dependency path.

**Outputs:** Updated `src/interactors/api/auth.py`

### TASK-10.3 — Rate limit inference requests per user
Add rate limiting middleware (e.g. `slowapi`) to cap inference requests per user per minute. Return HTTP 429 with `Retry-After` when the limit is exceeded.

**Outputs:** Updated `src/interactors/api/app.py`, new rate limit config

---

## EPIC-11: Fast E2E Tests

> Re-enable the E2E test suite on CI/CD without slowing down every PR.

### TASK-11.1 — Add scheduled E2E workflow
Add `.github/workflows/e2e.yml` triggered on `schedule: cron` (once daily at 02:00 UTC) and on `workflow_dispatch`. Run `pytest tests/e2e/` against the deployed environment with appropriate secrets.

**Outputs:** `.github/workflows/e2e.yml`

### TASK-11.2 — Fix or skip currently broken E2E tests
Audit `tests/e2e/` and either fix broken tests or mark them `@pytest.mark.skip(reason="...")` with a tracking note. Ensure the suite passes cleanly in the scheduled run.

**Outputs:** Updated `tests/e2e/` files

