"""CLI: generate synthetic training and eval datasets."""

from __future__ import annotations

import argparse
import sys
import tempfile
import uuid
from pathlib import Path

from adapters.storage.local import LocalStorageAdapter
from adapters.storage.paths import dataset_eval_key, dataset_train_key
from adapters.storage.s3 import S3StorageAdapter
from domain.ports import StoragePort
from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE, generate


def _build_storage(backend: str, data_dir: str) -> StoragePort:
    if backend == "s3":
        return S3StorageAdapter()
    return LocalStorageAdapter(base_dir=Path(data_dir))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic llm-api training data.")
    parser.add_argument("--data-dir", default="data", help="Base dir for local storage (default: data)")
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--storage", choices=["local", "s3"], default="s3", help="Storage backend (default: s3)")
    parser.add_argument("--dataset-id", default=None, help="Dataset ID (default: auto-generated UUID)")
    args = parser.parse_args(argv)

    dataset_id = args.dataset_id or str(uuid.uuid4())
    storage = _build_storage(args.storage, args.data_dir)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        ok = generate(
            data_dir=staging,
            train_size=args.train_size,
            eval_size=args.eval_size,
            seed=args.seed,
        )
        if not ok:
            sys.exit(1)

        train_key = dataset_train_key(dataset_id)
        eval_key = dataset_eval_key(dataset_id)
        storage.upload(staging / "train.jsonl", train_key)
        storage.upload(staging / "eval.jsonl", eval_key)

    print(f"dataset_id: {dataset_id}")
    print(f"  train: {train_key}")
    print(f"  eval:  {eval_key}")
    sys.exit(0)


if __name__ == "__main__":
    main()
