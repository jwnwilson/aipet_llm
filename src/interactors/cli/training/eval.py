"""CLI: evaluate a fine-tuned checkpoint against the pet-brain eval set.

Thin interactor — all ML logic lives in domain.train.evaluate.
Writes eval_results.json with {"valid_pct": float, "passed": bool}.

Exit codes:
  0  — eval passed (valid_pct >= PASS_THRESHOLD)
  1  — eval ran but failed threshold
  2+ — unexpected error (import failure, missing file, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned HF checkpoint against the pet-brain eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--eval-data", required=True, dest="eval_data",
        help="Path to eval.jsonl.",
    )
    parser.add_argument(
        "--output", default="eval_results.json",
        help="Path to write eval_results.json.",
    )
    args = parser.parse_args(argv)

    try:
        from domain.train.evaluate import evaluate, infer_hf, load_hf_pipeline
    except ImportError as exc:
        print(f"ERROR: missing dependency — {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        pipe = load_hf_pipeline(args.checkpoint)
        exit_code, valid_pct = evaluate(
            Path(args.eval_data),
            lambda prompt: infer_hf(pipe, prompt),
        )
    except Exception as exc:
        print(f"ERROR: eval failed — {exc}", file=sys.stderr)
        sys.exit(2)

    passed = exit_code == 0
    result = {"valid_pct": valid_pct, "passed": passed}
    Path(args.output).write_text(json.dumps(result))
    print(f"valid_pct={valid_pct:.3f}  passed={passed}", flush=True)
    sys.exit(exit_code)  # 0 = passed, 1 = below threshold


if __name__ == "__main__":
    main()
