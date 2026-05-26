"""Unit tests for the RunPod bootstrap script.

Verifies the wheel install command includes [training] extras so that
transformers, datasets, accelerate, etc. are present inside the pod.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")
os.environ.setdefault("RUN_ID", "runpod/test-run-abc123")


def _make_s3_mock(whl_key: str = "runpod/test-run-abc123/llm_api-0.1.0-py3-none-any.whl") -> MagicMock:
    """Return a boto3 S3 mock that serves a single .whl key."""
    s3 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": whl_key}]}
    ]
    s3.get_paginator.return_value = paginator
    s3.download_file.return_value = None
    s3.put_object.return_value = None
    return s3


def _run_bootstrap(s3: MagicMock, monkeypatch) -> list[list]:
    """Run bootstrap.main() with mocked S3 and subprocess; return subprocess call list."""
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("RUN_ID", "runpod/test-run-abc123")

    subprocess_calls: list[list] = []

    def fake_subprocess_run(cmd, **kwargs):
        subprocess_calls.append(list(cmd))
        return MagicMock(returncode=0)

    with (
        patch("boto3.client", return_value=s3),
        patch("subprocess.run", side_effect=fake_subprocess_run),
        patch("runpy.run_module"),
    ):
        # Force reimport so module-level env vars are re-evaluated.
        sys.modules.pop("adapters.compute.runpod.bootstrap", None)
        from adapters.compute.runpod import bootstrap
        bootstrap.main()

    return subprocess_calls


class TestBootstrapWheelInstall:
    """The bootstrap must install the wheel with the [training] extras group."""

    def test_pip_install_includes_training_extras(self, monkeypatch):
        """Regression: pip install must use wheel[training], not bare wheel path.

        Root cause of 'transformers not installed' error: the [training] extras
        (transformers, datasets, accelerate, peft, bitsandbytes, sentencepiece)
        are optional in pyproject.toml and were not being installed inside the pod.
        """
        calls = _run_bootstrap(_make_s3_mock(), monkeypatch)

        assert calls, "Expected at least one subprocess.run call"
        install_arg = calls[0][-1]

        assert install_arg.endswith("[training]"), (
            f"pip install must target wheel[training] to install transformers and "
            f"other HuggingFace training deps. Got: {install_arg!r}"
        )

    def test_pip_install_wheel_path_is_under_tmp(self, monkeypatch):
        """The wheel must be downloaded to /tmp (the pod's writable scratch area)."""
        calls = _run_bootstrap(_make_s3_mock(), monkeypatch)

        install_arg = calls[0][-1]
        raw_path = install_arg.removesuffix("[training]")

        assert raw_path.startswith("/tmp"), (
            f"Wheel should be downloaded to /tmp. Got path: {raw_path!r}"
        )

    def test_status_set_to_pending_at_start(self, monkeypatch):
        """S3 status.txt must be written as 'pending' before pip install begins."""
        s3 = _make_s3_mock()
        _run_bootstrap(s3, monkeypatch)

        put_calls = [c for c in s3.put_object.call_args_list]
        assert put_calls, "Expected at least one s3.put_object call"
        first_call_kwargs = put_calls[0].kwargs
        assert first_call_kwargs["Body"] == b"pending"
        assert first_call_kwargs["Key"].endswith("/status.txt")

    def test_exits_with_error_when_no_wheel_in_s3(self, monkeypatch):
        """Bootstrap must exit non-zero when no .whl is found in S3."""
        s3 = _make_s3_mock()
        # Override paginator to return no .whl keys.
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": [{"Key": "runpod/test-run/data/train.jsonl"}]}]
        s3.get_paginator.return_value = paginator

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
        monkeypatch.setenv("RUN_ID", "runpod/test-run-abc123")

        with (
            patch("boto3.client", return_value=s3),
            patch("runpy.run_module"),
        ):
            sys.modules.pop("adapters.compute.runpod.bootstrap", None)
            from adapters.compute.runpod import bootstrap
            with pytest.raises(SystemExit):
                bootstrap.main()

    def test_delegates_to_remote_worker_not_training_script(self, monkeypatch):
        """Bootstrap must invoke remote_worker, not the old per-adapter training_script."""
        mock_run_module = MagicMock()
        s3 = _make_s3_mock()
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
        monkeypatch.setenv("RUN_ID", "runpod/test-run-abc123")

        with (
            patch("boto3.client", return_value=s3),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch("runpy.run_module", mock_run_module),
        ):
            sys.modules.pop("adapters.compute.runpod.bootstrap", None)
            from adapters.compute.runpod import bootstrap
            bootstrap.main()

        called_module = mock_run_module.call_args[0][0]
        assert called_module == "interactors.cli.training.remote_worker", (
            f"Expected remote_worker, got: {called_module!r}"
        )
