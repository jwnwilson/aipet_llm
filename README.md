# AI Pet LLM (JSON contract + SFT)

Train a **small causal LM** (default **DistilGPT2**) to map **game context JSON** → **PetTurn JSON**, with **Pydantic validation**, **CPU-oriented inference**, and optional **OpenAI-compatible teacher labeling**.

## Contract

- **Input**: `GameContext` — pet, needs (0–100), scene, `constraints.allowed_actions_by_need`, and `task`.
- **Output**: `PetTurn` — `decision` (primary need + legal action), `needs_after_intent`, `emotion`, `dialog`, `confidence`.

See `data/raw/example.jsonl` for one full row. Export JSON Schema:

```bash
uv sync
uv run python scripts/export_schema.py --which both
```

## Setup (uv)

```bash
uv sync
```

## Data format (JSONL)

One JSON object per line:

```json
{"context": { ... GameContext ... }, "output": { ... PetTurn ... }}
```

Aliases: `input` / `target` (string or object) are accepted in `prepare_dataset.py`.

## Validate

```bash
uv run python scripts/validate_dataset.py --input data/raw/example.jsonl
```

## Train / val split

```bash
uv run python scripts/prepare_dataset.py \
  --input data/raw/example.jsonl \
  --train-output data/processed/train.jsonl \
  --val-output data/processed/val.jsonl \
  --val-ratio 0.15
```

With only one example, validation still passes; add more rows before serious training.

## Supervised fine-tuning (CPU training is slow; GPU recommended for iteration)

```bash
uv run python scripts/train_sft.py \
  --config configs/sft.yaml \
  --train-file data/processed/train.jsonl \
  --val-file data/processed/val.jsonl
```

Checkpoints and the merged model live under `train.output_dir` from `configs/sft.yaml` (default `models/aipet-sft/`). The last export is `.../final/`.

## Inference (CPU)

```bash
uv run python scripts/infer.py \
  --model-dir models/aipet-sft/final \
  --context data/raw/example_context.json
```

`--context` must be a **single JSON object** file (not JSONL); see `data/raw/example_context.json`. For a one-off:

```bash
uv run python scripts/infer.py --model-dir models/aipet-sft/final --context-json '{"pet":{"id":"x","name":"x","species":"fox","mood_hint":"ok"},"needs":{"hunger":50,"tiredness":50,"boredom":50,"social":50,"toilet":50},"scene":{"id":"s","tags":[],"objects":[],"players":[],"other_pets":[]},"constraints":{"allowed_actions_by_need":{"hunger":["forage"],"tiredness":["nap"],"boredom":["explore"],"social":["call attention"],"toilet":["find bush"]}},"task":"Pick ONE need and action."}'
```

Generation uses `configs/sft.yaml` → `infer.*`. If JSON is invalid, the script **retries once** with a stricter suffix, then returns a **small safe fallback** PetTurn (`confidence: 0`).

## Serve as API

Run a local HTTP API:

```bash
make serve MODEL_DIR=models/aipet-sft/final HOST=0.0.0.0 PORT=8000
```

If you trained a smoke checkpoint, use:

```bash
make serve MODEL_DIR=models/aipet-sft-smoke/final
```

Endpoints:

- `GET /health`
- `POST /v1/pet/turn`

Request body:

```json
{
  "context": { "... GameContext JSON ..." }
}
```

Response body:

- Valid `PetTurn` JSON, or
- fallback `PetTurn` with `"_warning": "used_fallback"` when generation fails contract checks.

## Teacher API (optional labeling)

Input JSONL: lines with `context` only (or `input`). Calls an OpenAI-compatible **chat completions** endpoint.

```bash
export OPENAI_API_KEY=...
uv run python scripts/teacher_label.py \
  --input data/raw/contexts_only.jsonl \
  --output data/raw/labeled_from_teacher.jsonl \
  --model gpt-4o-mini \
  --cache-file .cache/teacher_label_cache.json
```

Teacher labeling now supports response caching + retry backoff to reduce API cost and transient failures.

## Evaluation and runtime checks

Evaluate JSON quality + task alignment:

```bash
uv run python scripts/eval_infer.py \
  --model-dir models/aipet-sft/final \
  --input data/processed/val.jsonl
```

Measure runtime latency/fallback behavior:

```bash
uv run python scripts/runtime_check.py \
  --model-dir models/aipet-sft/final \
  --input data/processed/val.jsonl
```

## uv notes

- Lock dependencies: `uv lock`
- Include dev group: `uv sync --group dev`

Override base URL with `OPENAI_API_BASE` (default `https://api.openai.com/v1/chat/completions`). Rows that fail validation are skipped.

## Layout

| Path | Role |
|------|------|
| `src/aipet_distill/schemas.py` | Pydantic models |
| `src/aipet_distill/validate.py` | Allowlist + id checks |
| `src/aipet_distill/prompts.py` | Fixed tags + system text for train/infer |
| `configs/sft.yaml` | Model name, max length, Trainer + generation knobs |
| `scripts/train_sft.py` | Causal LM SFT with masked prompt loss |
| `scripts/infer.py` | Load HF model, JSON extract, validate, fallback |
| `scripts/eval_infer.py` | JSON parse/contract validity + action/need match metrics |
| `scripts/runtime_check.py` | End-to-end CPU latency and fallback-rate check |

## Notes

- **Pet `target` ids**: `type: "pet"` must reference `scene.other_pets[].id` (not the controlled pet).
- Long JSON contexts are prompt-truncated automatically to fit small model token limits.
- For production CPU speed, consider **quantization** or **ONNX** after you are happy with quality; this repo stays minimal.
- With **hundreds** of examples, prioritize **diversity** across needs and scenes; use `teacher_label.py` to paraphrase or expand contexts while re-validating outputs.
