# EPIC-7: Rename `aipet` → `llm-api` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every `aipet`-prefixed identifier, string, and resource name across the codebase, k8s manifests, Terraform, and CI/CD with `llm-api` equivalents. No new files; all existing tests must pass at the end.

**Architecture:** Three-phase scripted rename — (1) `git mv` for directory and example-file renames, (2) ordered `find | xargs sed` sweeps for bulk string replacement, (3) targeted `perl` pass for bare `aipet` words left in docstrings. Three prerequisites must be completed outside the PR (GitHub secrets, Auth0 action, Terraform apply).

**Tech Stack:** Bash, sed, perl, git mv, Terraform, Kubernetes YAML, Python

---

## Prerequisites (complete before writing any code)

These actions affect live infrastructure and must happen before the PR is merged so CI doesn't break mid-flight.

### 1 — Create new GitHub secrets

In the GitHub repo → Settings → Secrets, create these secrets with the **same values** as their `AIPET_` counterparts:

| New secret | Copy value from |
|---|---|
| `LLM_API_AWS_ACCESS_KEY_ID` | `LLM_API_AWS_ACCESS_KEY_ID` |
| `LLM_API_AWS_SECRET_ACCESS_KEY` | `LLM_API_AWS_SECRET_ACCESS_KEY` |
| `LLM_API_DB_PASSWORD` | `LLM_API_DB_PASSWORD` |

Keep the old secrets in place until after the first successful CI run with the new names.

### 2 — Update Auth0 custom action

In the Auth0 Dashboard → Actions → Flows → Login, find the action that sets custom claims.  
Change the claim key from `https://aipet/roles` to `https://llm-api/roles`.  
Deploy the action. This must be live before the code change deploys.

### 3 — Apply Terraform to create new ECR repos

The `variables.tf` default for `repo_name` changes from `llm-api` → `llm-api`, and `ecr_temporal_ui` repo_name changes from `llm-api-temporal-ui` → `llm-api-temporal-ui`. Terraform will destroy and recreate both ECR repos — pick a low-traffic window.

```bash
cd infra/terraform
terraform plan   # review: expects destroy+create for ecr and ecr_temporal_ui
terraform apply
```

After apply, note the new ECR registry URLs from `terraform output`.

---

## File Map

All files modified, none created:

| File | Change |
|---|---|
| `infra/k8s/llm-api/` | Renamed to `infra/k8s/llm-api/` |
| `infra/k8s/llm-api/aipet-secrets.secret.example.yaml` | Renamed to `llm-api-secrets.secret.example.yaml` |
| `infra/k8s/llm-api/llm-api-db.secret.example.yaml` | Renamed to `llm-api-db.secret.example.yaml` |
| `models/kaggle_kernels/aipet-*/` | Renamed to `llm-api-*/` (local dirs only, not Kaggle slug IDs) |
| All `*.py`, `*.yml`, `*.yaml`, `*.toml`, `*.tf`, `*.md`, `*.json` | Bulk sed sweep |

---

## Task 1: Rename directories and example files

**Files:**
- Rename: `infra/k8s/llm-api/` → `infra/k8s/llm-api/`
- Rename: `infra/k8s/llm-api/aipet-secrets.secret.example.yaml` → `llm-api-secrets.secret.example.yaml`
- Rename: `infra/k8s/llm-api/llm-api-db.secret.example.yaml` → `llm-api-db.secret.example.yaml`
- Rename: `models/kaggle_kernels/aipet-*/` → `models/kaggle_kernels/llm-api-*/`

- [ ] **Step 1: Rename the k8s directory**

```bash
git mv infra/k8s/llm-api infra/k8s/llm-api
```

- [ ] **Step 2: Rename example secret files inside the new directory**

```bash
git mv infra/k8s/llm-api/aipet-secrets.secret.example.yaml \
       infra/k8s/llm-api/llm-api-secrets.secret.example.yaml
git mv infra/k8s/llm-api/llm-api-db.secret.example.yaml \
       infra/k8s/llm-api/llm-api-db.secret.example.yaml
```

- [ ] **Step 3: Rename Kaggle kernel directories**

```bash
cd models/kaggle_kernels
git mv aipet-fast-test          llm-api-fast-test
git mv aipet-fast-test-data     llm-api-fast-test-data
git mv aipet-fast-test-eval     llm-api-fast-test-eval
git mv aipet-fast-test-eval-output llm-api-fast-test-eval-output
git mv aipet-v3                 llm-api-v3
git mv aipet-v3-data            llm-api-v3-data
cd ../..
```

- [ ] **Step 4: Verify git status — only renames, nothing deleted**

```bash
git status --short | head -30
```

Expected: all lines start with `R` (renamed). No `D` (deleted) lines.

- [ ] **Step 5: Commit directory renames**

```bash
git add -A
git commit -m "refactor: rename k8s and kaggle dirs aipet → llm-api"
```

---

## Task 2: Bulk sed sweep — Round 1 (compound strings, longest first)

Run all commands from the repo root. The order matters: longer/more-specific strings must be replaced before the shorter substrings they contain.

**Files:** All `*.py`, `*.yml`, `*.yaml`, `*.toml`, `*.tf`, `*.md`, `*.json` files (excluding `.git/`, `node_modules/`, `ui/node_modules/`, `.terraform/`).

- [ ] **Step 1: Define the file-finder command**

```bash
FIND='find . \
  -not -path "./.git/*" \
  -not -path "./node_modules/*" \
  -not -path "./ui/node_modules/*" \
  -not -path "*/.terraform/*" \
  -not -path "./data/*" \
  -not -name "kernel-metadata.json" \
  \( -name "*.py" -o -name "*.yml" -o -name "*.yaml" \
     -o -name "*.toml" -o -name "*.tf" -o -name "*.md" \
     -o -name "*.json" -o -name "*.sh" \) \
  -print0'
```

- [ ] **Step 2: Replace compound `aipet-*` strings (longest first)**

```bash
eval "$FIND" | xargs -0 sed -i '' 's/llm-api/llm-api/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-secrets/llm-api-secrets/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-db-secret/llm-api-db-secret/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-temporal-ui/llm-api-temporal-ui/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-tls/llm-api-tls/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-training/llm-api-training/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-kaggle-e2e/llm-api-kaggle-e2e/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-my-exp/llm-api-my-exp/g'
```

- [ ] **Step 3: Replace shorter `aipet-*` strings**

```bash
eval "$FIND" | xargs -0 sed -i '' 's/llm-api/llm-api/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-db/llm-api-db/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm-api-exp/llm-api-exp/g'
```

- [ ] **Step 4: Replace filenames and path segments**

```bash
eval "$FIND" | xargs -0 sed -i '' 's/aipet\.gguf/model.gguf/g'
eval "$FIND" | xargs -0 sed -i '' 's/llm_api_bootstrap/llm_api_bootstrap/g'
eval "$FIND" | xargs -0 sed -i '' 's/aipet\.db/llm-api.db/g'
```

- [ ] **Step 5: Replace path and env-var patterns**

```bash
eval "$FIND" | xargs -0 sed -i '' 's|~/llm-api|~/llm-api|g'
eval "$FIND" | xargs -0 sed -i '' \
  's/LLM_API_AWS_ACCESS_KEY_ID/LLM_API_AWS_ACCESS_KEY_ID/g'
eval "$FIND" | xargs -0 sed -i '' \
  's/LLM_API_AWS_SECRET_ACCESS_KEY/LLM_API_AWS_SECRET_ACCESS_KEY/g'
eval "$FIND" | xargs -0 sed -i '' \
  's/LLM_API_DB_PASSWORD/LLM_API_DB_PASSWORD/g'
eval "$FIND" | xargs -0 sed -i '' \
  's/LLM_API_TEST_MODEL_PATH/LLM_API_TEST_MODEL_PATH/g'
```

- [ ] **Step 6: Spot-check a few key files**

```bash
grep -n "aipet" \
  infra/k8s/llm-api/deployment.yaml \
  infra/k8s/llm-api/postgres.yaml \
  .github/workflows/deploy.yml \
  src/interactors/temporal/worker.py \
  pyproject.toml
```

Expected output: only postgres internal user/db (`postgres-user: "aipet"`, `postgres-db: "aipet"`) and the `llm-api-db.secret.yaml` comment in the example file — everything else should be gone.

---

## Task 3: Sed sweep — Round 2 (URL namespaces, Python expression patterns, bare word)

- [ ] **Step 1: Replace Auth0 JWT claim namespace and test audience**

```bash
eval "$FIND" | xargs -0 sed -i '' \
  's|https://aipet/|https://llm-api/|g'
eval "$FIND" | xargs -0 sed -i '' \
  's|https://api\.aipet\.|https://api.llm-api.|g'
```

Verify:
```bash
grep -rn "https://aipet" src/ tests/
```
Expected: no output.

- [ ] **Step 2: Replace Python expression patterns**

```bash
# SSH adapter tmux session prefix: f"aipet-{...}" → f"llm-api-{...}"
eval "$FIND" | xargs -0 sed -i '' \
  's/f"aipet-{/f"llm-api-{/g'

# Temporal default experiment name: or "aipet" → or "llm-api"
eval "$FIND" | xargs -0 sed -i '' \
  's/or "aipet"/or "llm-api"/g'

# Colab Google token path
eval "$FIND" | xargs -0 sed -i '' \
  's|config/aipet/|config/llm-api/|g'
```

- [ ] **Step 3: Replace remaining bare `aipet` words in docstrings and comments**

```bash
perl -pi -e 's/\baipet\b/llm-api/g' \
  src/interactors/api/app.py \
  src/interactors/api/routes/inference.py \
  src/adapters/compute/ssh.py \
  src/adapters/inference.py \
  tests/integration/test_api.py \
  tests/e2e/test_model_quality.py \
  tests/e2e/test_real_inference.py \
  tests/e2e/test_inference_behaviour.py \
  infra/k8s/llm-api/llm-api-db.secret.example.yaml
```

- [ ] **Step 4: Full grep to confirm no aipet remains (outside intentional exceptions)**

```bash
grep -rn "aipet" . \
  --include="*.py" --include="*.yml" --include="*.yaml" \
  --include="*.toml" --include="*.tf" --include="*.md" \
  --include="*.json" --include="*.sh" \
  | grep -v ".git" \
  | grep -v "kernel-metadata.json" \
  | grep -v "postgres-user.*aipet" \
  | grep -v "postgres-db.*aipet" \
  | grep -v "database-url.*:aipet:" \
  | grep -v "CHANGE_ME"
```

Expected: **no output**. The only allowed survivors are the postgres internal user/db name inside the DB secret files (a DB migration is out of scope for this rename).

If any unexpected matches appear, add a targeted sed to eliminate them and re-run.

- [ ] **Step 5: Commit the sweep changes**

```bash
git add -A
git commit -m "refactor: bulk rename aipet → llm-api across all files"
```

---

## Task 4: Run the test suite and verify

- [ ] **Step 1: Run unit tests**

```bash
uv run pytest tests/unit/ -v
```

Expected: all tests pass. If any test fails with a string mismatch (e.g. a mock expecting `"llm-api-training"` but receiving `"llm-api-training"`), trace to the file and add a targeted sed fix — do not manually patch if the bulk sweep missed it; add the missing sed to Task 3 and re-run.

- [ ] **Step 2: Run integration tests**

```bash
uv run pytest tests/integration/ -v
```

Expected: all tests pass (integration tests use a real SQLite DB; the default DB path `llm-api.db` will be created fresh).

- [ ] **Step 3: Commit if tests pass**

```bash
git add -A
git commit -m "test: verify tests pass after aipet → llm-api rename"
```

---

## Post-merge checklist (manual, after CI deploys)

- [ ] Delete old `llm-api` and `llm-api-temporal-ui` ECR repos from AWS console (after first successful push to `llm-api` and `llm-api-temporal-ui`)
- [ ] Run k8s cleanup:
  ```bash
  kubectl delete daemonset llm-api
  kubectl delete statefulset llm-api-db
  kubectl delete service llm-api llm-api-db
  kubectl delete secret llm-api-secrets llm-api-db-secret
  kubectl delete ingress llm-api
  ```
- [ ] Add DNS record `llm-api.jwnwilson.co.uk` → ingress IP (cert-manager will issue a new TLS cert automatically)
- [ ] Remove old DNS record `llm-api.jwnwilson.co.uk` once the new one resolves
- [ ] Update `CORS_ORIGINS` in the k8s secret if the UI references the old API domain
- [ ] Delete old GitHub secrets (`LLM_API_AWS_ACCESS_KEY_ID`, `LLM_API_AWS_SECRET_ACCESS_KEY`, `LLM_API_DB_PASSWORD`) after CI succeeds with new names
- [ ] If you have a local `models/test_model.gguf`, rename it: `mv models/test_model.gguf models/test_model.gguf`
