"""Supervised fine-tuning (causal LM) on context -> PetTurn JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from aipet_distill.jsonl import iter_jsonl
from aipet_distill.prompts import build_prompt_text
from aipet_distill.validate import validate_turn


def _load_rows(path: Path) -> list[dict]:
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
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


class JsonSftDataset(Dataset):
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
            raise ValueError("No samples after tokenization; increase max_length or shorten JSON.")

    def _tokenize_row(self, ctx: dict, out: dict) -> dict[str, list[int]] | None:
        resp = json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n"
        prompt = build_prompt_text(ctx)
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
            keep_prompt = self.max_length - len(r_ids)
            p_ids = p_ids[-keep_prompt:]
        labels = [-100] * len(p_ids) + r_ids
        return {"input_ids": p_ids + r_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    ap.add_argument("--train-file", type=Path, required=True)
    ap.add_argument("--val-file", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None, help="Override config train.output_dir")
    ap.add_argument("--max-steps", type=int, default=None, help="Smoke test: cap total optimizer steps")
    ap.add_argument("--no-truncate-prompt", action="store_true", help="Drop rows that exceed max_length")
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
    val_rows = _load_rows(args.val_file)
    ds_train = JsonSftDataset(train_rows, tokenizer, max_length, truncate_prompt=not args.no_truncate_prompt)
    ds_val = JsonSftDataset(val_rows, tokenizer, max_length, truncate_prompt=not args.no_truncate_prompt)

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
            seq_len = len(f["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
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
