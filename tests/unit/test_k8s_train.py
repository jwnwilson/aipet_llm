"""Unit tests for the k8s_train entrypoint (train-only, no eval)."""
from __future__ import annotations

import importlib


def test_module_importable_and_exposes_run() -> None:
    mod = importlib.import_module("interactors.cli.training.k8s_train")
    assert callable(getattr(mod, "run", None)), "run() must be defined and callable"


def test_old_module_removed() -> None:
    """k8s_train_eval has been replaced by k8s_train; guard against accidental re-addition."""
    import importlib.util
    spec = importlib.util.find_spec("interactors.cli.training.k8s_train_eval")
    assert spec is None, (
        "interactors.cli.training.k8s_train_eval still exists — "
        "it was replaced by k8s_train (eval runs on the Temporal worker)"
    )
