"""SFT causal LM to generate only the spoken line (SAY) given simple text + chosen need/action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from aipet_distill.jsonl import iter_jsonl
from aipet_distill.simple_format import build_say_prompt, game_context_to_simple_text
from aipet_distill.validate import validate_turn


def _load_rows(path: Path, require_non_empty: bool = True) -> list[dict]:
    rows: list[dict] = []
    for obj in iter_jsonl(path):
        ctx = obj.get("context") or obj.get("input")
        out = obj.get("output") or obj.get("target")
        if ctx is None or out is None:
            continue
        if isinstance(out, str):
            out = json.loads(out)
        errs = validate_turn(ctx, out)
        if errs:
            raise ValueError(f"{path}: invalid row: {errs}")
        rows.append({"context": ctx, "output": out})
    if not rows and require_non_empty:
        raise ValueError(f"{path}: no rows")
    return rows


class SaySftDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int, truncate_prompt: bool = True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncate_prompt = truncate_prompt
        self.samples: list[dict[str, list[int]]] = []
        for row in rows:
            item = self._tokenize_row(row["context"], row["output"])
            if item is not None:
                self.samples.append(item)
        if not self.samples:
            raise ValueError("No samples after tokenization; increase max_length.")

    def _tokenize_row(self, ctx: dict, out: dict) -> dict[str, list[int]] | None:
        dec = out.get("decision") or {}
        need = dec.get("primary_need")
        action = (dec.get("action") or {}).get("name")
        say = (out.get("dialog") or "").strip()
        if not need or not action or not say:
            return None
        simple = game_context_to_simple_text(ctx)
        prompt = build_say_prompt(simple, need, action)
        resp = say + "\n"
        p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        r_ids = self.tokenizer.encode(resp, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            r_ids = r_ids + [eos]
        if len(r_ids) >= self.max_length:
            return None
        if len(p_ids) + len(r_ids) > self.max_length:
            if not self.truncate_prompt:
                return None
            keep = self.max_length - len(r_ids)
            p_ids = p_ids[-keep:]
        labels = [-100] * len(p_ids) + r_ids
        return {"input_ids": p_ids + r_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/two_stage.yaml"))
    ap.add_argument("--train-file", type=Path, required=True)
    ap.add_argument("--val-file", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-truncate-prompt", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = cfg["model_name"]
    max_length = int(cfg["max_length"])
    train_cfg = dict(cfg["train"])
    if "evaluation_strategy" in train_cfg and "eval_strategy" not in train_cfg:
        train_cfg["eval_strategy"] = train_cfg.pop("evaluation_strategy")
    if args.output_dir is not None:
        train_cfg["output_dir"] = str(args.output_dir)
    if args.max_steps is not None:
        train_cfg["max_steps"] = int(args.max_steps)
        train_cfg.pop("num_train_epochs", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = _load_rows(args.train_file)
    val_rows = _load_rows(args.val_file, require_non_empty=False)
    if not val_rows:
        val_rows = train_rows[:1]
    ds_train = SaySftDataset(train_rows, tokenizer, max_length, truncate_prompt=not args.no_truncate_prompt)
    ds_val = SaySftDataset(val_rows, tokenizer, max_length, truncate_prompt=not args.no_truncate_prompt)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise SystemExit("Tokenizer has no pad_token_id")

    def data_collator(features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, f in enumerate(features):
            L = len(f["input_ids"])
            input_ids[i, :L] = torch.tensor(f["input_ids"], dtype=torch.long)
            labels[i, :L] = torch.tensor(f["labels"], dtype=torch.long)
            attention_mask[i, :L] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    model = AutoModelForCausalLM.from_pretrained(model_name)
    targs = TrainingArguments(**train_cfg)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
    )
    trainer.train()
    final_dir = Path(targs.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
