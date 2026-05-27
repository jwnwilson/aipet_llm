"""Unit tests for the RunPod bootstrap script.

Verifies:
- Wheel install command includes [training] extras.
- Restart-loop prevention: idempotency guard + self-termination.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")
os.environ.setdefault("RUN_ID", "runpod/test-run-abc123")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_s3_mock(
    whl_key: str = "runpod/test-run-abc123/llm_api-0.1.0-py3-none-any.whl",
    existing_status: str | None = None,
) -> MagicMock:
    """Return a boto3 S3 mock that serves a single .whl key.

    Args:
        existing_status: if set, ``get_object`` returns this value for status.txt
                         (simulates a pod that was restarted after completing).
                         If None, ``get_object`` raises to simulate no status.txt.
    """
    s3 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": whl_key}]}
    ]
    s3.get_paginator.return_value = paginator
    s3.download_file.return_value = None
    s3.put_object.return_value = None

    if existing_status is not None:
        body_mock = MagicMock()
        body_mock.read.return_value = existing_status.encode()
        s3.get_object.return_value = {"Body": body_mock}
    else:
        # Simulate missing status.txt — _read_existing_status must return None.
        s3.get_object.side_effect = Exception("NoSuchKey")

    return s3


def _run_bootstrap(
    s3: MagicMock,
    monkeypatch,
    *,
    pod_id: str = "",
    runpod_api_key: str = "",
) -> tuple[list[list], MagicMock, MagicMock]:
    """Run bootstrap.main() with mocked S3, subprocess, and runpod.

    Returns:
        (subprocess_calls, mock_run_module, mock_runpod_module)
    """
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("RUN_ID", "runpod/test-run-abc123")
    if pod_id:
        monkeypatch.setenv("RUNPOD_POD_ID", pod_id)
    else:
        monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    if runpod_api_key:
        monkeypatch.setenv("RUNPOD_API_KEY", runpod_api_key)
    else:
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    subprocess_calls: list[list] = []

    def fake_subprocess_run(cmd, **kwargs):
        subprocess_calls.append(list(cmd))
        return MagicMock(returncode=0)

    mock_run_module = MagicMock()
    mock_runpod = MagicMock()

    with (
        patch("boto3.client", return_value=s3),
        patch("subprocess.run", side_effect=fake_subprocess_run),
        patch("runpy.run_module", mock_run_module),
    ):
        sys.modules["runpod"] = mock_runpod
        # Force reimport so module-level env vars are re-evaluated.
        sys.modules.pop("adapters.compute.runpod.bootstrap", None)
        from adapters.compute.runpod import bootstrap
        bootstrap.main()

    return subprocess_calls, mock_run_module, mock_runpod


# ---------------------------------------------------------------------------
# Wheel install behaviour (unchanged from before)
# ---------------------------------------------------------------------------

class TestBootstrapWheelInstall:
    """The bootstrap must install the wheel with the [training] extras group."""

    def test_pip_install_includes_training_extras(self, monkeypatch):
        """Regression: pip install must use wheel[training], not bare wheel path.

        Root cause of 'transformers not installed' error: the [training] extras
        (transformers, datasets, accelerate, peft, bitsandbytes, sentencepiece)
        are optional in pyproject.toml and were not being installed inside the pod.
        """
        calls, _, _ = _run_bootstrap(_make_s3_mock(), monkeypatch)

        assert calls, "Expected at least one subprocess.run call"
        install_arg = calls[0][-1]

        assert install_arg.endswith("[training]"), (
            f"pip install must target wheel[training] to install transformers and "
            f"other HuggingFace training deps. Got: {install_arg!r}"
        )

    def test_pip_install_wheel_path_is_under_tmp(self, monkeypatch):
        """The wheel must be downloaded to /tmp (the pod's writable scratch area)."""
        calls, _, _ = _run_bootstrap(_make_s3_mock(), monkeypatch)

        install_arg = calls[0][-1]
        raw_path = install_arg.removesuffix("[training]")

        assert raw_path.startswith("/tmp"), (
            f"Wheel should be downloaded to /tmp. Got path: {raw_path!r}"
        )

    def test_status_set_to_pending_at_start(self, monkeypatch):
        """S3 status.txt must be written as 'pending' before pip install begins."""
        s3 = _make_s3_mock()  # get_object raises → no prior status
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
        monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

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
        _, mock_run_module, _ = _run_bootstrap(_make_s3_mock(), monkeypatch)

        called_module = mock_run_module.call_args[0][0]
        assert called_module == "interactors.cli.training.remote_worker", (
            f"Expected remote_worker, got: {called_module!r}"
        )


# ---------------------------------------------------------------------------
# Idempotency guard (restart-loop prevention)
# ---------------------------------------------------------------------------

class TestBootstrapIdempotencyGuard:
    """On pod restart, bootstrap must detect a completed run and not re-run it."""

    def test_skips_job_when_status_is_done(self, monkeypatch):
        """If status.txt == 'done', bootstrap must not write 'pending' or re-run."""
        s3 = _make_s3_mock(existing_status="done")
        _, mock_run_module, _ = _run_bootstrap(s3, monkeypatch)

        # Must not overwrite the finished status with "pending"
        pending_writes = [
            c for c in s3.put_object.call_args_list
            if c.kwargs.get("Body") == b"pending"
        ]
        assert not pending_writes, (
            "Bootstrap must not write 'pending' when run already completed ('done')"
        )

        # Must not re-run the training job
        mock_run_module.assert_not_called()

    def test_skips_job_when_status_is_failed(self, monkeypatch):
        """If status.txt == 'failed', bootstrap must not write 'pending' or re-run."""
        s3 = _make_s3_mock(existing_status="failed")
        _, mock_run_module, _ = _run_bootstrap(s3, monkeypatch)

        pending_writes = [
            c for c in s3.put_object.call_args_list
            if c.kwargs.get("Body") == b"pending"
        ]
        assert not pending_writes, (
            "Bootstrap must not write 'pending' when run already completed ('failed')"
        )
        mock_run_module.assert_not_called()

    def test_proceeds_normally_when_status_is_running(self, monkeypatch):
        """If status.txt == 'running' (pod killed mid-job), bootstrap should proceed."""
        # 'running' means the pod was preempted — re-running is the right behaviour.
        s3 = _make_s3_mock(existing_status="running")
        _, mock_run_module, _ = _run_bootstrap(s3, monkeypatch)

        mock_run_module.assert_called_once()

    def test_proceeds_normally_when_no_status_txt_exists(self, monkeypatch):
        """Fresh run (no prior status.txt) must proceed as usual."""
        s3 = _make_s3_mock()  # get_object raises → _read_existing_status returns None
        _, mock_run_module, _ = _run_bootstrap(s3, monkeypatch)

        mock_run_module.assert_called_once()


# ---------------------------------------------------------------------------
# Self-termination
# ---------------------------------------------------------------------------

class TestBootstrapSelfTermination:
    """After job completion, bootstrap must terminate the RunPod pod."""

    def test_terminates_pod_after_successful_job(self, monkeypatch):
        """_self_terminate must be called with the correct pod_id on success."""
        _, _, mock_runpod = _run_bootstrap(
            _make_s3_mock(),
            monkeypatch,
            pod_id="pod-abc123",
            runpod_api_key="rp-fake-key",
        )

        mock_runpod.terminate_pod.assert_called_once_with("pod-abc123")

    def test_sets_api_key_before_terminating(self, monkeypatch):
        """RUNPOD_API_KEY from env must be assigned to the runpod module before terminating."""
        _, _, mock_runpod = _run_bootstrap(
            _make_s3_mock(),
            monkeypatch,
            pod_id="pod-abc123",
            runpod_api_key="rp-my-key-xyz",
        )

        assert mock_runpod.api_key == "rp-my-key-xyz"

    def test_terminates_pod_on_idempotency_bypass(self, monkeypatch):
        """When skipping a completed run, bootstrap must still self-terminate."""
        s3 = _make_s3_mock(existing_status="done")
        _, _, mock_runpod = _run_bootstrap(
            s3,
            monkeypatch,
            pod_id="pod-restarted-xyz",
            runpod_api_key="rp-fake-key",
        )

        mock_runpod.terminate_pod.assert_called_once_with("pod-restarted-xyz")

    def test_does_not_raise_when_pod_id_missing(self, monkeypatch):
        """Missing RUNPOD_POD_ID must not crash — just skip termination gracefully."""
        _, _, mock_runpod = _run_bootstrap(
            _make_s3_mock(),
            monkeypatch,
            pod_id="",
            runpod_api_key="",
        )

        mock_runpod.terminate_pod.assert_not_called()

    def test_does_not_raise_when_terminate_api_fails(self, monkeypatch):
        """A RunPod API error during termination must not mask the job result."""
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
        monkeypatch.setenv("RUN_ID", "runpod/test-run-abc123")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc123")
        monkeypatch.setenv("RUNPOD_API_KEY", "rp-fake-key")

        mock_runpod = MagicMock()
        mock_runpod.terminate_pod.side_effect = RuntimeError("API timeout")
        sys.modules["runpod"] = mock_runpod

        s3 = _make_s3_mock()
        # Should complete without raising despite the API failure.
        with (
            patch("boto3.client", return_value=s3),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch("runpy.run_module"),
        ):
            sys.modules.pop("adapters.compute.runpod.bootstrap", None)
            from adapters.compute.runpod import bootstrap
            bootstrap.main()  # must not raise
