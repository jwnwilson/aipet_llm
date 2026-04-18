#!/usr/bin/env python3
"""Split validated context/output JSONL into train and validation files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from aipet_distill.jsonl import iter_jsonl, write_jsonl
from aipet_distill.validate import validate_turn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="JSONL with context + output per line")
    p.add_argument("--train-output", type=Path, required=True)
    p.add_argument("--val-output", type=Path, required=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    errors = 0
    for obj in iter_jsonl(args.input):
        ctx = obj.get("context") or obj.get("input")
        out = obj.get("output") or obj.get("target")
        if ctx is None or out is None:
            print("skip: missing context/input or output/target keys")
            errors += 1
            continue
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except json.JSONDecodeError as e:
                print(f"skip: bad output JSON string: {e}")
                errors += 1
                continue
        errs = validate_turn(ctx, out)
        if errs:
            print(f"skip invalid: {errs}")
            errors += 1
            continue
        rows.append({"context": ctx, "output": out})

    if not rows:
        raise SystemExit("No valid rows; fix data or validation rules.")

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio)) if len(rows) >= 2 else 0
    if len(rows) == 1:
        n_val = 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    if not train_rows and val_rows:
        train_rows = val_rows[:-1]
        val_rows = val_rows[-1:]

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.val_output, val_rows)
    print(f"train={len(train_rows)} val={len(val_rows)} skipped={errors}")


if __name__ == "__main__":
    main()
