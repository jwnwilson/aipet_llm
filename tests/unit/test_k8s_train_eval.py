"""Smoke-level unit test for the k8s_train_eval interactor module."""
from __future__ import annotations

import importlib


def test_module_importable_and_exposes_run() -> None:
    mod = importlib.import_module("interactors.cli.training.k8s_train_eval")
    assert callable(getattr(mod, "run", None)), "run() must be defined and callable"
