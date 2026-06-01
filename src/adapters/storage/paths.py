"""Canonical S3 key constructors for model artefacts.

All code that writes or reads a model from storage must go through these
helpers so that the key format is consistent across export, activation,
and inference pod injection.
"""

from __future__ import annotations


def standalone_model_key(model_id: str, model_name: str = "model") -> str:
    """Return the S3 key for a named / standalone model export.

    Format: model/{model_id}/{model_name}.gguf
    """
    return f"model/{model_id}/{model_name}.gguf"


def workflow_model_key(workflow_id: str, model_name: str = "model") -> str:
    """Return the S3 key for a model produced by a training workflow.

    Format: workflow/{workflow_id}/model/{model_name}.gguf
    """
    return f"workflow/{workflow_id}/model/{model_name}.gguf"


def dataset_train_key(dataset_id: str) -> str:
    """Return the S3 key for a dataset's training split.

    Format: dataset/{dataset_id}/train.jsonl
    """
    return f"dataset/{dataset_id}/train.jsonl"


def dataset_eval_key(dataset_id: str) -> str:
    """Return the S3 key for a dataset's evaluation split.

    Format: dataset/{dataset_id}/eval.jsonl
    """
    return f"dataset/{dataset_id}/eval.jsonl"
