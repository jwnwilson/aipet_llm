"""Two-stage inference: classify need||action, then LM generates SAY. Output line: need|action|say."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from aipet_distill.simple_format import (
    LABEL_SEP,
    build_say_prompt,
    clamp_need_action,
    format_line,
    game_context_to_simple_text,
    validate_simple_turn,
)


def predict_need_action(pipeline: Any, context: dict[str, Any]) -> tuple[str, str]:
    text = game_context_to_simple_text(context)
    label = pipeline.predict([text])[0]
    parts = str(label).split(LABEL_SEP, 1)
    if len(parts) != 2:
        return clamp_need_action(context, "boredom", "explore")
    need, action = parts[0].strip(), parts[1].strip()
    return clamp_need_action(context, need, action)


def generate_say(
    tokenizer: Any,
    model: Any,
    prompt: str,
    infer_cfg: dict[str, Any],
    max_new_tokens: int,
    max_positions: int,
) -> str:
    max_prompt_tokens = max(64, max_positions - max_new_tokens)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if infer_cfg.get("do_sample", False):
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(infer_cfg.get("temperature", 0.3))
        gen_kwargs["top_p"] = float(infer_cfg.get("top_p", 0.9))
    else:
        gen_kwargs["do_sample"] = False
        if infer_cfg.get("num_beams"):
            gen_kwargs["num_beams"] = int(infer_cfg["num_beams"])
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    say = raw.strip().split("\n")[0].strip()
    return say


def run_two_stage(
    context: dict[str, Any],
    classifier: Any,
    tokenizer: Any,
    model: Any,
    cfg: dict[str, Any],
) -> tuple[str, str, str, str]:
    infer_cfg = cfg.get("infer", {})
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    max_positions = int(getattr(model.config, "n_positions", 1024))

    need, action = predict_need_action(classifier, context)
    errs = validate_simple_turn(context, need, action)
    if errs:
        need, action = clamp_need_action(context, need, action)
    simple = game_context_to_simple_text(context)
    prompt = build_say_prompt(simple, need, action)
    say = generate_say(tokenizer, model, prompt, infer_cfg, max_new_tokens, max_positions)

    line = format_line(need, action, say or "* ... *")
    return need, action, say, line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/two_stage.yaml"))
    ap.add_argument("--classifier", type=Path, default=Path("models/action_classifier.joblib"))
    ap.add_argument("--say-model-dir", type=Path, required=True, help="HF folder from train_say_sft final/")
    ap.add_argument("--context", type=Path, help="JSON file with GameContext")
    ap.add_argument("--context-json", type=str)
    args = ap.parse_args()

    if not args.classifier.exists():
        raise SystemExit(f"Classifier not found: {args.classifier}")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if args.context:
        context = json.loads(args.context.read_text(encoding="utf-8"))
    elif args.context_json:
        context = json.loads(args.context_json)
    else:
        raise SystemExit("Provide --context or --context-json")

    clf = joblib.load(args.classifier)
    tokenizer = AutoTokenizer.from_pretrained(args.say_model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.say_model_dir)
    model.eval()

    _, _, _, line = run_two_stage(context, clf, tokenizer, model, cfg)
    print(line)


if __name__ == "__main__":
    main()
