# Game AI Starter — Small Specialised LLM

A minimal Python project for fine-tuning and deploying a fast, CPU-friendly
language model to power game NPCs.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
# 1. install deps (creates .venv from pyproject.toml)
uv sync

# 2. generate synthetic training data
uv run python -m src.dataset

# 3. fine-tune with LoRA  (~5 min on GPU, ~30 min CPU for 135M model)
uv run python -m src.train

# 4. run inference demo
uv run python -m src.inference
```

## Project layout

```
game_ai_starter/
├── data/               # saved HuggingFace datasets
├── models/             # LoRA adapter checkpoints
├── src/
│   ├── config.py       # all hyperparams in one place
│   ├── dataset.py      # synthetic data generator (replace with real logs)
│   ├── train.py        # LoRA fine-tuning loop
│   └── inference.py    # GameAI class with LRU cache
├── tests/
│   └── test_dataset.py
└── pyproject.toml      # deps + packaging
```

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Base model | SmolLM2-135M | 270MB on disk, fast CPU inference |
| Fine-tuning | LoRA (r=8) | Trains in minutes, adapter is ~10MB |
| Inference | Greedy decoding | Deterministic + fastest path |
| Caching | `lru_cache(512)` | Repeated NPC triggers → ~0ms |

## Swapping the base model

Change `base_model` in `src/config.py`. Tested options:

- `HuggingFaceTB/SmolLM2-135M` — smallest, fastest
- `HuggingFaceTB/SmolLM2-360M` — better quality, still CPU-friendly
- `microsoft/phi-3-mini-4k-instruct` — best quality, needs ~2GB RAM

## Next steps

- Replace `generate_samples()` in `dataset.py` with real game logs
- Add INT4 quantisation: `uv pip install bitsandbytes` and set `load_in_4bit=True`
- Export to GGUF via `llama.cpp` for the lowest possible CPU latency
