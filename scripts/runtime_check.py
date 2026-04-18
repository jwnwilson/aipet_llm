#!/usr/bin/env python3
"""Run CPU latency/fallback checks over many contexts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from statistics import mean

from aipet_distill.jsonl import iter_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="JSONL with context or context+output")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    latencies: list[float] = []
    total = 0
    fallback = 0

    for row in iter_jsonl(args.input):
        if args.limit and total >= args.limit:
            break
        ctx = row.get("context") or row.get("input")
        if ctx is None:
            continue
        total += 1
        started = time.perf_counter()
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/infer.py",
                "--model-dir",
                str(args.model_dir),
                "--context-json",
                json.dumps(ctx, ensure_ascii=False),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)
        if '"warning": "used_fallback"' in proc.stderr:
            fallback += 1

    if total == 0:
        raise SystemExit("No contexts found.")

    sorted_l = sorted(latencies)
    p95_idx = max(0, int(0.95 * len(sorted_l)) - 1)
    report = {
        "rows": total,
        "fallback_rate_pct": round((fallback / total) * 100.0, 2),
        "latency_ms_mean": round(mean(latencies), 2),
        "latency_ms_p95": round(sorted_l[p95_idx], 2),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
