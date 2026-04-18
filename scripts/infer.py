#!/usr/bin/env python3
"""CPU-friendly inference: context JSON -> PetTurn JSON with retry + fallback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from aipet_distill.constants import NEED_KEYS
from aipet_distill.prompts import build_prompt_text
from aipet_distill.validate import validate_turn


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return None
    return None


def fallback_turn(context: dict[str, Any]) -> dict[str, Any]:
    needs = (context.get("needs") or {}).copy()
    for k in NEED_KEYS:
        needs.setdefault(k, 50)
    return {
        "decision": {
            "primary_need": "boredom",
            "action": {"need": "boredom", "name": "explore", "target": None},
        },
        "needs_after_intent": {k: int(needs.get(k, 50)) for k in NEED_KEYS},
        "emotion": {
            "label": "neutral",
            "intensity": 0.35,
            "reason": "Model output invalid; safe fallback.",
        },
        "dialog": "*looks around curiously*",
        "confidence": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    ap.add_argument("--model-dir", type=Path, required=True, help="HF model folder (e.g. models/.../final)")
    ap.add_argument("--context", type=Path, help="JSON file with a single GameContext object")
    ap.add_argument("--context-json", type=str, help="Inline JSON string (alternative to --context)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    infer_cfg = cfg.get("infer", {})
    max_new_tokens = int(cfg.get("max_new_tokens", 256))

    if args.context:
        context = json.loads(args.context.read_text(encoding="utf-8"))
    elif args.context_json:
        context = json.loads(args.context_json)
    else:
        raise SystemExit("Provide --context FILE or --context-json STRING")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.eval()
    max_positions = int(getattr(model.config, "n_positions", 1024))

    def generate(extra_suffix: str = "") -> str:
        prompt = build_prompt_text(context) + extra_suffix
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_positions - max_new_tokens)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if infer_cfg.get("do_sample", True):
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(infer_cfg.get("temperature", 0.2))
            gen_kwargs["top_p"] = float(infer_cfg.get("top_p", 0.9))
        else:
            gen_kwargs["do_sample"] = False
            if infer_cfg.get("num_beams"):
                gen_kwargs["num_beams"] = int(infer_cfg["num_beams"])
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    text = generate()
    obj = extract_json_object(text)
    if obj is None or validate_turn(context, obj):
        strict = (
            "Return ONLY one JSON object. No markdown. Keys: decision, needs_after_intent, "
            "emotion, dialog, confidence.\n"
        )
        text2 = generate(extra_suffix=strict)
        obj = extract_json_object(text2)

    if obj is None or validate_turn(context, obj):
        obj = fallback_turn(context)
        print(
            json.dumps({"warning": "used_fallback", "raw_first": text[:500]}, ensure_ascii=False),
            file=sys.stderr,
        )
    print(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
