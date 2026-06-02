"""Unit tests for the k8s_train entrypoint (train-only, no eval)."""
from __future__ import annotations

import importlib


def test_k8s_train_module_removed() -> None:
    """k8s_train has been consolidated into remote_worker; guard against re-addition."""
    spec = importlib.util.find_spec("interactors.cli.training.k8s_train")
    assert spec is None, (
        "interactors.cli.training.k8s_train still exists — "
        "it was consolidated into remote_worker (use JOB_TYPE=train)"
    )


def test_old_module_removed() -> None:
    """k8s_train_eval has been replaced by k8s_train; guard against accidental re-addition."""
    import importlib.util
    spec = importlib.util.find_spec("interactors.cli.training.k8s_train_eval")
    assert spec is None, (
        "interactors.cli.training.k8s_train_eval still exists — "
        "it was replaced by k8s_train (eval runs on the Temporal worker)"
    )
