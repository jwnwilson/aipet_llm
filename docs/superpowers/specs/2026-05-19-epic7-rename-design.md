# EPIC-7: Rename `aipet` → `llm-api` Design

**Date:** 2026-05-19

## Summary

Full rename of all `aipet`-prefixed identifiers, resource names, and configuration values to `llm-api` equivalents across the entire codebase, infrastructure manifests, and Terraform. The UI integration (TASK-7.1.2) is already complete — `ui/` exists, `deploy-ui.yml` is wired, and Terraform modules are in place. This spec covers TASK-7.1.1 only.

---

## Rename Mapping

| Old | New | Layer |
|-----|-----|-------|
| `aipet-llm` (k8s resource/label names) | `llm-api` | k8s YAMLs |
| `aipet-llm-secrets` | `llm-api-secrets` | k8s Secret |
| `aipet-db` | `llm-api-db` | k8s StatefulSet + Service |
| `aipet-db-secret` | `llm-api-db-secret` | k8s Secret |
| `aipet-llm-api` | `llm-api` | `pyproject.toml` |
| `aipet-llm:latest` | `llm-api:latest` | `docker-compose.yml` |
| `aipet` (docker-compose service name) | `llm-api` | `docker-compose.yml` |
| `aipet-training` (Temporal task queue) | `llm-api-training` | Python source |
| `aipet.gguf` | `model.gguf` | Python source + k8s env var |
| `~/aipet` (SSH/remote work dir) | `~/llm-api` | SSH adapter + remote compute |
| `infra/k8s/aipet-llm/` | `infra/k8s/llm-api/` | directory |
| ECR repo `aipet-llm` | `llm-api` | Terraform + `deploy.yml` |
| ECR repo `aipet-temporal-ui` | `llm-api-temporal-ui` | `deploy.yml` env var |
| `models/kaggle_kernels/aipet-*/` | `models/kaggle_kernels/llm-api-*/` | local dirs only |

**Kaggle kernel `id` fields** inside `kernel-metadata.json` are left unchanged — those are fixed Kaggle account slugs that cannot be renamed.

---

## Implementation Approach

**Bulk sed sweep + manual exceptions.**

1. `git mv infra/k8s/aipet-llm infra/k8s/llm-api` — rename the k8s directory.
2. A scoped `find | xargs sed` sweep replaces all `aipet` variants in Python source, YAML, TOML, Terraform, and doc files.
3. Three edge cases handled manually after the sweep (see below).
4. Kaggle kernel directories renamed with `git mv`.

---

## Edge Cases

### ECR Repository

AWS ECR repos cannot be renamed. A new `llm-api` repo must be created via Terraform before `deploy.yml` can push to it.

- Add a new `aws_ecr_repository "llm_api"` resource to `infra/terraform/main.tf` (or the ECR module).
- Update `ECR_REPOSITORY` in `.github/workflows/deploy.yml` from `aipet-llm` to `llm-api`.
- Update `ECR_TEMPORAL_UI_REPOSITORY` from `aipet-temporal-ui` to `llm-api-temporal-ui`.
- Update the hardcoded secret names in the `deploy.yml` sync steps: `aipet-llm-secrets` → `llm-api-secrets`, `aipet-db-secret` → `llm-api-db-secret`.
- After the first successful deploy to the new repos, delete the old `aipet-llm` and `aipet-temporal-ui` ECR repos manually from the AWS console.
- No IAM changes needed — the existing OIDC role policy grants access to all ECR repos in the account.

### Temporal Task Queue

Renaming `aipet-training` → `llm-api-training` orphans any in-flight Temporal workflows. This change must be applied when no workflows are running. The rename touches `worker.py`, `workflows.py`, `activities.py`, and `trigger_training.py`.

### Model Filename

`aipet.gguf` → `model.gguf` is updated in Python source and the k8s `MODEL_PATH` env var. Any GGUF already uploaded to S3 under the old key continues to work until the next training run uploads under the new key. No migration of existing S3 objects is required.

---

## k8s Live Migration

**Strategy: PR-only, let CI apply.**

When the PR merges, `deploy.yml` applies the renamed manifests. Kubernetes creates the new `llm-api` resources alongside the existing `aipet-llm` ones. After rollout succeeds, run the following manual cleanup:

```bash
kubectl delete daemonset aipet-llm
kubectl delete statefulset aipet-db
kubectl delete service aipet-llm aipet-db
kubectl delete secret aipet-llm-secrets aipet-db-secret
```

The secret sync step in `deploy.yml` uses `--dry-run=client -o yaml | kubectl apply -f -`, so it handles the new secret names (`llm-api-secrets`, `llm-api-db-secret`) cleanly on first apply.

**Note:** The `deploy.yml` sed command that substitutes `<ECR_REPOSITORY_URL>` must be updated to reference the new manifest path (`infra/k8s/llm-api/deployment.yaml` etc.).

---

## Files Created / Modified

| Path | Action |
|------|--------|
| `infra/k8s/llm-api/` | Renamed from `infra/k8s/aipet-llm/` |
| `infra/k8s/llm-api/*.yaml` | Updated — all `aipet-llm*` / `aipet-db*` resource names replaced |
| `infra/terraform/main.tf` | Modified — add new `llm-api` ECR repo resource |
| `.github/workflows/deploy.yml` | Modified — `ECR_REPOSITORY: llm-api`, updated k8s manifest paths |
| `pyproject.toml` | Modified — `name = "llm-api"` |
| `docker-compose.yml` | Modified — service name, image name, `MODEL_PATH` |
| `src/interactors/temporal/worker.py` | Modified — task queue name |
| `src/interactors/temporal/workflows.py` | Modified — task queue name |
| `src/interactors/temporal/activities.py` | Modified — any `aipet` strings |
| `src/interactors/cli/training/trigger_training.py` | Modified — task queue name |
| `src/adapters/compute/ssh.py` | Modified — remote work dir |
| `src/adapters/compute/runpod/` | Modified — any `aipet` strings |
| `src/adapters/compute/vastai/` | Modified — any `aipet` strings |
| `src/adapters/compute/kaggle/` | Modified — any `aipet` strings |
| `src/adapters/inference.py` | Modified — default model path |
| Python source (all remaining) | Modified — `aipet.gguf` → `model.gguf`, `~/aipet` → `~/llm-api` |
| `models/kaggle_kernels/llm-api-*/` | Renamed from `models/kaggle_kernels/aipet-*/` |
| `README.md`, `docs/**` | Modified — prose references updated |
| `CLAUDE.md` | Modified — project name updated |

---

## Post-Deploy Checklist

- [ ] `kubectl delete` old `aipet-llm` resources (see k8s migration section)
- [ ] Delete old `aipet-llm` ECR repo from AWS console after first successful push to `llm-api`
- [ ] Confirm no Temporal workflows are running before deploying (queue name change)
- [ ] Update `UI_BUCKET` and `UI_CF_DISTRIBUTION_ID` GitHub secrets if terraform apply changes outputs
