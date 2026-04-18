#!/usr/bin/env python3
"""Print JSON Schema for GameContext and PetTurn (stdout)."""

from __future__ import annotations

import argparse
import json

from aipet_distill.schemas import GameContext, PetTurn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--which", choices=("game", "pet", "both"), default="both")
    args = p.parse_args()
    if args.which in ("game", "both"):
        print(json.dumps(GameContext.model_json_schema(), indent=2))
    if args.which == "both":
        print("---")
    if args.which in ("pet", "both"):
        print(json.dumps(PetTurn.model_json_schema(), indent=2))


if __name__ == "__main__":
    main()
