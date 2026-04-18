"""Evaluate inference quality/latency against a labeled JSONL dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ctx = obj.get("context") or obj.get("input")
            out = obj.get("output") or obj.get("target")
            if isinstance(out, str):
                out = json.loads(out)
            if ctx is None or out is None:
                raise ValueError(f"{path}:{i}: missing context/output")
            yield i, ctx, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="Labeled JSONL with context+output")
    ap.add_argument("--limit", type=int, default=0, help="Optional max rows")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    infer_cfg = cfg.get("infer", {})
    max_new_tokens = int(cfg.get("max_new_tokens", 256))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.eval()
    max_positions = int(getattr(model.config, "n_positions", 1024))

    total = 0
    json_parsed = 0
    contract_valid = 0
    primary_need_match = 0
    action_name_match = 0
    latencies_ms: list[float] = []

    def generate(context: dict[str, Any]) -> tuple[str, float]:
        prompt = build_prompt_text(context)
        max_prompt_tokens = max(64, max_positions - max_new_tokens)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": bool(infer_cfg.get("do_sample", True)),
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(infer_cfg.get("temperature", 0.2))
            gen_kwargs["top_p"] = float(infer_cfg.get("top_p", 0.9))
        elif infer_cfg.get("num_beams"):
            gen_kwargs["num_beams"] = int(infer_cfg["num_beams"])

        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True), elapsed_ms

    for _, ctx, gold in iter_rows(args.input):
        if args.limit and total >= args.limit:
            break
        total += 1
        raw_text, latency_ms = generate(ctx)
        latencies_ms.append(latency_ms)
        pred = extract_json_object(raw_text)
        if pred is not None:
            json_parsed += 1
            errs = validate_turn(ctx, pred)
            if not errs:
                contract_valid += 1
                if pred["decision"]["primary_need"] == gold["decision"]["primary_need"]:
                    primary_need_match += 1
                if pred["decision"]["action"]["name"] == gold["decision"]["action"]["name"]:
                    action_name_match += 1

    if total == 0:
        raise SystemExit("No rows evaluated.")

    def pct(n: int) -> float:
        return (100.0 * n) / total

    report = {
        "rows": total,
        "json_parse_rate_pct": round(pct(json_parsed), 2),
        "contract_valid_rate_pct": round(pct(contract_valid), 2),
        "primary_need_match_pct": round(pct(primary_need_match), 2),
        "action_name_match_pct": round(pct(action_name_match), 2),
        "latency_ms_mean": round(mean(latencies_ms), 2),
        "latency_ms_p95": round(sorted(latencies_ms)[max(0, int(0.95 * len(latencies_ms)) - 1)], 2),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
