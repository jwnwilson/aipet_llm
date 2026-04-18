#!/usr/bin/env python3
"""Validate JSONL rows (context + output) against the game contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aipet_distill.jsonl import iter_jsonl
from aipet_distill.validate import validate_turn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()

    bad = 0
    good = 0
    for i, obj in enumerate(iter_jsonl(args.input), 1):
        ctx = obj.get("context") or obj.get("input")
        out = obj.get("output") or obj.get("target")
        if ctx is None or out is None:
            print(f"line {i}: missing context or output")
            bad += 1
            continue
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except json.JSONDecodeError as e:
                print(f"line {i}: bad output JSON: {e}")
                bad += 1
                continue
        errs = validate_turn(ctx, out)
        if errs:
            print(f"line {i}: {errs}")
            bad += 1
        else:
            good += 1
    print(f"ok={good} bad={bad}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
