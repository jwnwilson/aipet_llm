from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from aipet_distill.constants import NEED_KEYS
from aipet_distill.prompts import build_prompt_text
from aipet_distill.validate import validate_turn


class TurnRequest(BaseModel):
    context: dict[str, Any]


class ServerConfig(BaseModel):
    config_path: Path = Field(default=Path("configs/sft.yaml"))
    model_dir: Path = Field(default=Path("models/aipet-sft/final"))


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
    for key in NEED_KEYS:
        needs.setdefault(key, 50)
    return {
        "decision": {
            "primary_need": "boredom",
            "action": {"need": "boredom", "name": "explore", "target": None},
        },
        "needs_after_intent": {key: int(needs.get(key, 50)) for key in NEED_KEYS},
        "emotion": {
            "label": "neutral",
            "intensity": 0.35,
            "reason": "Model output invalid; safe fallback.",
        },
        "dialog": "*looks around curiously*",
        "confidence": 0.0,
    }


app = FastAPI(title="AI Pet LLM API", version="0.1.0")

_server_cfg = ServerConfig(
    config_path=Path(os.environ.get("AIPET_CONFIG_PATH", "configs/sft.yaml")),
    model_dir=Path(os.environ.get("AIPET_MODEL_DIR", "models/aipet-sft/final")),
)
if not _server_cfg.config_path.exists():
    raise RuntimeError(f"Config file not found: {_server_cfg.config_path}")
if not _server_cfg.model_dir.exists():
    raise RuntimeError(f"Model directory not found: {_server_cfg.model_dir}")
_cfg = yaml.safe_load(_server_cfg.config_path.read_text(encoding="utf-8"))
_infer_cfg = _cfg.get("infer", {})
_max_new_tokens = int(_cfg.get("max_new_tokens", 256))

_tokenizer = AutoTokenizer.from_pretrained(_server_cfg.model_dir, use_fast=True)
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token
_model = AutoModelForCausalLM.from_pretrained(_server_cfg.model_dir)
_model.eval()
_max_positions = int(getattr(_model.config, "n_positions", 1024))


def _generate(context: dict[str, Any], extra_suffix: str = "") -> str:
    prompt = build_prompt_text(context) + extra_suffix
    max_prompt_tokens = max(64, _max_positions - _max_new_tokens)
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    gen_kwargs = {
        "max_new_tokens": _max_new_tokens,
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    if _infer_cfg.get("do_sample", True):
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(_infer_cfg.get("temperature", 0.2))
        gen_kwargs["top_p"] = float(_infer_cfg.get("top_p", 0.9))
    else:
        gen_kwargs["do_sample"] = False
        if _infer_cfg.get("num_beams"):
            gen_kwargs["num_beams"] = int(_infer_cfg["num_beams"])
    with torch.inference_mode():
        out = _model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return _tokenizer.decode(gen_ids, skip_special_tokens=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/pet/turn")
def pet_turn(req: TurnRequest) -> dict[str, Any]:
    context = req.context
    raw = _generate(context)
    obj = extract_json_object(raw)
    if obj is None or validate_turn(context, obj):
        strict = (
            "Return ONLY one JSON object. No markdown. Keys: decision, needs_after_intent, "
            "emotion, dialog, confidence.\n"
        )
        raw_retry = _generate(context, extra_suffix=strict)
        obj = extract_json_object(raw_retry)

    if obj is None or validate_turn(context, obj):
        out = fallback_turn(context)
        out["_warning"] = "used_fallback"
        return out

    errs = validate_turn(context, obj)
    if errs:
        raise HTTPException(status_code=422, detail={"errors": errs})
    return obj

