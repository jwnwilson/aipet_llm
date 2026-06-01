"""CLI: convert a HuggingFace checkpoint to a quantised GGUF for RPi deployment.

Thin interactor — all conversion logic lives in domain.train.export.

Exit codes:
  0  — GGUF created successfully
  1  — llama.cpp setup issue (convert script or quantize binary missing)
  2+ — unexpected error
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a HF checkpoint to a quantised GGUF via llama.cpp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the quantised GGUF file.",
    )
    parser.add_argument(
        "--quantize", default="Q4_K_M",
        help="Quantisation type (e.g. Q4_K_M, Q8_0, F16).",
    )
    parser.add_argument(
        "--llama-cpp-dir", default=None, dest="llama_cpp_dir",
        help="Path to llama.cpp directory (uses domain default if omitted).",
    )
    args = parser.parse_args(argv)

    try:
        from domain.train.export import export
    except ImportError as exc:
        print(f"ERROR: missing dependency — {exc}", file=sys.stderr)
        sys.exit(2)

    kwargs: dict = {}
    if args.llama_cpp_dir:
        kwargs["llama_cpp_dir"] = Path(args.llama_cpp_dir)

    try:
        export(
            checkpoint=Path(args.checkpoint),
            output=Path(args.output),
            quantize=args.quantize,
            **kwargs,
        )
    except SystemExit:
        raise  # llama.cpp setup failures call sys.exit — propagate the code
    except Exception as exc:
        print(f"ERROR: export failed — {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
