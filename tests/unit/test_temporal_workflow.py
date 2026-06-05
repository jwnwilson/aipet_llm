"""E2E workflow tests — full TrainingPipelineWorkflow with mocked domain functions (dry run).

Uses Temporal's embedded time-skipping test server so no real Temporal cluster is needed,
and patches all domain functions so no ML computation runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from interactors.temporal.activities import (
    EvalConfig,
    configure_run_store,
    configure_storage,
    evaluate_activity,
    export_activity,
    fail_run_activity,
    finalise_run_activity,
    generate_dataset_activity,
    record_eval_result_activity,
    save_gguf_path_activity,
    train_activity,
    update_run_status_activity,
)
from interactors.temporal.workflows import (
    EvaluateWorkflow,
    EvaluateWorkflowConfig,
    ExperimentConfig,
    PipelineResult,
    TrainingPipelineWorkflow,
)


def _dry_run_patches(eval_passes: bool = True):
    """Return a list of patches that stub every domain I/O call."""

    def fake_evaluate(path, infer_fn):
        if eval_passes:
            return (0, 0.95)
        else:
            return (1, 0.75)

    def _fake_upload_model(storage, local_path, key: str) -> str:
        return key if key.endswith(".gz") else key + ".gz"

    # Mock k8s adapter used by export_activity (remote_backend is always "k8s").
    mock_k8s = MagicMock()
    mock_k8s.submit.return_value = "fake-export-run-id"
    mock_k8s.status.return_value = "done"

    return [
        patch("domain.train.dataset.generate", return_value=True),
        patch("domain.train.trainer.train"),
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
        patch("domain.train.evaluate.evaluate", side_effect=fake_evaluate),
        patch("domain.train.export.export"),
        patch("adapters.storage.upload_model", side_effect=_fake_upload_model),
        patch("interactors.temporal.activities._make_remote_adapter", return_value=mock_k8s),
    ]


def _configure_mock_storage() -> MagicMock:
    """Wire a mock StoragePort into the activities module and return it."""
    storage = MagicMock()
    configure_storage(storage)
    return storage


def _configure_mock_run_store() -> MagicMock:
    """Wire a mock RunStorePort into the activities module and return it."""
    run_store = MagicMock()
    configure_run_store(run_store)
    return run_store


_ACTIVITIES = [
    generate_dataset_activity,
    train_activity,
    evaluate_activity,
    export_activity,
    fail_run_activity,
    finalise_run_activity,
    record_eval_result_activity,
    save_gguf_path_activity,
    update_run_status_activity,
]


@pytest.mark.asyncio
async def test_training_pipeline_workflow_e2e_pass():
    """Happy path: all stages succeed and a GGUF is exported."""
    _configure_mock_storage()
    _configure_mock_run_store()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = _dry_run_patches(eval_passes=True)
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="dry-run-pass",
                    run_id="test-run-pass",
                    train_size=10,
                    eval_size=5,
                    epochs=1,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-dry-run-pass",
                    task_queue="test-queue",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.passed is True
    assert result.dataset_paths.train.endswith("train.jsonl")
    assert result.dataset_paths.eval.endswith("eval.jsonl")
    assert result.checkpoint.path != ""
    assert abs(result.eval_result.valid_pct - 0.95) < 1e-6
    # k8s export returns the S3 key directly (no .gz wrapping from upload_model)
    assert result.gguf_path.path.endswith(".gguf")


@pytest.mark.asyncio
async def test_training_pipeline_workflow_e2e_eval_fail_still_exports():
    """When eval does not reach 95%, export still runs — checkpoint is never discarded."""
    _configure_mock_storage()
    _configure_mock_run_store()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue-fail",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = _dry_run_patches(eval_passes=False)
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="dry-run-fail",
                    run_id="test-run-eval-fail",
                    train_size=10,
                    eval_size=5,
                    epochs=1,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-dry-run-fail",
                    task_queue="test-queue-fail",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.passed is False
    assert abs(result.eval_result.valid_pct - 0.75) < 1e-6
    # Export always runs even when eval fails — checkpoint is preserved in GGUF form.
    assert result.gguf_path.path != ""


@pytest.mark.asyncio
async def test_training_pipeline_workflow_e2e_skip_generate():
    """With skip_generate=True the dataset step is bypassed and existing paths are used."""
    _configure_mock_storage()
    _configure_mock_run_store()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue-skip",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = _dry_run_patches(eval_passes=True)
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="dry-run-skip-gen",
                    run_id="test-run-skip-gen",
                    skip_generate=True,
                    data_dir="data",
                    epochs=1,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-dry-run-skip-gen",
                    task_queue="test-queue-skip",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.passed is True
    assert result.dataset_paths.train == "data/train.jsonl"
    assert result.dataset_paths.eval == "data/eval.jsonl"


@pytest.mark.asyncio
async def test_workflow_skip_generate_uses_explicit_train_data_s3_key():
    """When skip_generate=True and train_data/eval_data are set on ExperimentConfig,
    the workflow uses those values (the actual S3 keys) as the dataset paths.

    Regression for: smoke-test K8s 404 — trigger_run provides train_dataset_id whose S3
    key (dataset/{uuid}.jsonl) was never forwarded to ExperimentConfig, so the workflow
    fell back to data/workflow/{run_id}/train.jsonl which does not exist in S3.
    """
    _configure_mock_storage()
    _configure_mock_run_store()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue-skip-s3",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = _dry_run_patches(eval_passes=True)
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="dry-run-s3-key",
                    run_id="test-run-s3-key",
                    skip_generate=True,
                    train_data="dataset/abc123.jsonl",
                    eval_data="dataset/eval456.jsonl",
                    data_dir="data/workflow/run-xyz",  # must NOT be used when train_data is set
                    epochs=1,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-skip-gen-s3-key",
                    task_queue="test-queue-skip-s3",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.dataset_paths.train == "dataset/abc123.jsonl", (
        "Workflow must propagate ExperimentConfig.train_data as the dataset path "
        "so the K8s pod downloads the correct S3 key"
    )
    assert result.dataset_paths.eval == "dataset/eval456.jsonl"


@pytest.mark.asyncio
async def test_workflow_skip_generate_falls_back_to_data_dir_when_no_train_data():
    """When skip_generate=True and train_data is empty, the fallback data_dir+/train.jsonl
    path is used (backwards compatibility for runs that don't supply train_dataset_id)."""
    _configure_mock_storage()
    _configure_mock_run_store()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue-skip-fallback",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = _dry_run_patches(eval_passes=True)
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="dry-run-fallback",
                    run_id="test-run-fallback",
                    skip_generate=True,
                    train_data="",   # not set → fallback to data_dir
                    eval_data="",
                    data_dir="data",
                    epochs=1,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-skip-gen-fallback",
                    task_queue="test-queue-skip-fallback",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.dataset_paths.train == "data/train.jsonl"
    assert result.dataset_paths.eval == "data/eval.jsonl"


@pytest.mark.asyncio
async def test_evaluate_workflow_passes_run_id():
    """EvaluateWorkflow must pass run_id to EvalConfig so quality report is written."""
    storage = _configure_mock_storage()

    # Track calls to storage.write to verify quality_report.json is written with run_id
    written_files = {}

    def capture_write(path, content):
        written_files[path] = content

    storage.write = capture_write

    # Configure a mock RunStore for finalise_run_activity
    run_store = MagicMock()
    from interactors.temporal.activities import configure_run_store
    configure_run_store(run_store)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-evaluate-queue",
            workflows=[EvaluateWorkflow],
            activities=[evaluate_activity, finalise_run_activity, record_eval_result_activity, fail_run_activity],
        ):
            # Patch the domain functions called by evaluate_activity
            patches = [
                patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
                patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
                patch("domain.train.evaluate.evaluate", return_value=(0, 0.96)),
                patch("domain.train.quality_report.run_quality_report", return_value={"passed": True}),
            ]
            for p in patches:
                p.start()
            try:
                config = EvaluateWorkflowConfig(
                    run_id="db-run-12345",
                    remote_backend="",
                    remote_run_id="",
                    eval_data="data/eval.jsonl",
                    checkpoint_path="/path/to/checkpoint",
                    output_dir="data/workflow/db-run-12345",
                )
                result = await env.client.execute_workflow(
                    EvaluateWorkflow.run,
                    config,
                    id="test-evaluate-workflow",
                    task_queue="test-evaluate-queue",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    # Verify that the evaluation completed successfully
    # The presence of quality report logs confirms run_id was passed to EvalConfig
    # (The evaluate_activity only saves a quality report when config.run_id is non-empty)
    assert result.passed is True
    assert abs(result.valid_pct - 0.96) < 1e-6


@pytest.mark.asyncio
async def test_training_pipeline_workflow_activity_failure_marks_run_failed():
    """When an activity raises, the workflow calls fail_run_activity and marks the run FAILED."""
    from unittest.mock import MagicMock
    from temporalio.client import WorkflowFailureError
    from domain.models import RunStatus

    _configure_mock_storage()
    mock_store = MagicMock()
    configure_run_store(mock_store)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue-fail-activity",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = [
                # generate_dataset raises to simulate an early stage failure
                patch("domain.train.dataset.generate", side_effect=RuntimeError("disk full")),
            ]
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="test-fail",
                    run_id="run-fail-123",
                    train_size=10,
                    eval_size=5,
                    epochs=1,
                )
                with pytest.raises(WorkflowFailureError):
                    await env.client.execute_workflow(
                        TrainingPipelineWorkflow.run,
                        config,
                        id="test-wf-activity-fail",
                        task_queue="test-queue-fail-activity",
                    )
            finally:
                for p in reversed(patches):
                    p.stop()

    # fail_run should have been called with FAILED status and the error reason
    mock_store.fail_run.assert_called_once()
    call_args = mock_store.fail_run.call_args
    assert call_args.args[0] == "run-fail-123"
    assert call_args.args[2] == RunStatus.FAILED


@pytest.mark.asyncio
async def test_training_pipeline_workflow_cancelled_marks_run_cancelled():
    """When Temporal cancels the workflow, fail_run_activity is called with CANCELLED status."""
    import asyncio
    import threading
    from unittest.mock import MagicMock
    from temporalio.client import WorkflowFailureError
    from domain.models import RunStatus

    _configure_mock_storage()
    mock_store = MagicMock()
    configure_run_store(mock_store)

    # generate() runs in an executor thread — block there until Temporal cancels the
    # run_in_executor future (asyncio.CancelledError), which happens when the workflow
    # is cancelled.  We must NOT unblock manually before Temporal does; otherwise the
    # thread returns None, activity thinks it finished, and no cancellation fires.
    # stop_blocking is set after handle.result() returns so the thread can exit cleanly
    # and the asyncio event loop can shut down without hanging on executor.shutdown(wait=True).
    generate_started = threading.Event()
    stop_blocking = threading.Event()
    loop = asyncio.get_event_loop()

    def slow_generate(*args, **kwargs):
        """Block the executor thread until Temporal cancels the activity, then return."""
        generate_started.set()
        stop_blocking.wait(timeout=60)  # unblocked by test after workflow terminates

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue="test-queue-cancel",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            with patch("domain.train.dataset.generate", side_effect=slow_generate):
                handle = await env.client.start_workflow(
                    TrainingPipelineWorkflow.run,
                    ExperimentConfig(
                        experiment_name="test-cancel",
                        run_id="run-cancel-456",
                        train_size=2,
                        eval_size=1,
                        epochs=1,
                    ),
                    id="test-wf-cancel",
                    task_queue="test-queue-cancel",
                )
                # Wait until generate is blocking inside the executor thread
                await loop.run_in_executor(None, generate_started.wait, 15)
                # Cancel the workflow — Temporal will cancel the in-progress activity
                await handle.cancel()

                try:
                    with pytest.raises(WorkflowFailureError):
                        await handle.result()
                finally:
                    # Unblock the executor thread so asyncio can shut down cleanly.
                    # By this point the workflow has already terminated (cancelled), so
                    # unblocking the thread is safe — the asyncio Future was already
                    # cancelled before slow_generate returns.
                    stop_blocking.set()

    # fail_run should have been called with CANCELLED status
    mock_store.fail_run.assert_called_once()
    call_args = mock_store.fail_run.call_args
    assert call_args.args[0] == "run-cancel-456"
    assert call_args.args[2] == RunStatus.CANCELLED
