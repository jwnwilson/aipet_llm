"""Training pipeline workflow — orchestrates dataset generation through GGUF export."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    import asyncio
    from temporalio.exceptions import is_cancelled_exception
    from domain.models import RunStatus
    from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE
    from domain.train.config import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_PATIENCE, DEFAULT_WARMUP_RATIO
    from interactors.temporal.activities import (
        CheckpointPath,
        DatasetConfig,
        DatasetPaths,
        EvalConfig,
        EvalResult,
        ExportConfig,
        GGUFPath,
        TrainConfig,
        create_inference_activity,
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


@dataclass
class ExperimentConfig:
    experiment_name: str = ""
    model_id: str = ""
    model_name: str = ""
    run_id: str = ""
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    skip_generate: bool = False
    dry_run: bool = False
    data_dir: str = "data"
    output_dir: str = DEFAULT_OUTPUT_DIR
    gguf_output: str = "models/model.gguf"
    model: str = DEFAULT_MODEL
    train_size: int = TRAIN_SIZE
    eval_size: int = EVAL_SIZE
    seed: int = SEED
    # "k8s", "kaggle", or "ssh" — controls where fine-tuning runs.
    remote_backend: str = ""
    # None = auto-detect based on model size; True = always QLoRA; False = never QLoRA.
    force_qlora: bool | None = None
    # Explicit S3 keys for train/eval data.  When non-empty and skip_generate=True these
    # are used directly instead of deriving paths from data_dir, so that a dataset
    # uploaded via POST /api/datasets (stored at dataset/{uuid}/train.jsonl) is correctly
    # forwarded to the remote training job rather than a non-existent local path.
    train_data: str = ""
    eval_data: str = ""
    # When True (default), create an inference record after export regardless of eval outcome.
    # Set to False to gate inference record creation on eval success.
    always_create_inference: bool = True


@dataclass
class PipelineResult:
    run_id: str = ""
    experiment_name: str = ""
    dataset_paths: DatasetPaths = field(default_factory=DatasetPaths)
    checkpoint: CheckpointPath = field(default_factory=CheckpointPath)
    eval_result: EvalResult = field(default_factory=EvalResult)
    eval_outcome: str = ""   # "succeeded" | "failed" | "" (before eval runs)
    gguf_path: GGUFPath = field(default_factory=GGUFPath)
    passed: bool = False


_RETRY = RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0)
_NO_RETRY = RetryPolicy(maximum_attempts=1)


@workflow.defn
class TrainingPipelineWorkflow:
    def __init__(self) -> None:
        self._failed = False

    @workflow.signal
    def WorkflowFailed(self) -> None:
        """Signal received when a caller marks this workflow as externally failed."""
        self._failed = True

    @workflow.run
    async def run(self, config: ExperimentConfig) -> PipelineResult:
        workflow.logger.info(
            "ExperimentConfig received: model=%s epochs=%d patience=%d warmup_ratio=%.4f "
            "remote_backend=%s skip_generate=%s",
            config.model, config.epochs, config.patience, config.warmup_ratio,
            config.remote_backend, config.skip_generate,
        )
        result = PipelineResult(run_id=config.run_id, experiment_name=config.experiment_name)

        try:
            if config.skip_generate:
                # Use explicit S3 keys when provided (e.g. uploaded via train_dataset_id).
                # Fall back to data_dir-derived paths for local runs or re-runs that
                # already have data at the standard location.
                result.dataset_paths = DatasetPaths(
                    train=config.train_data or f"{config.data_dir}/train.jsonl",
                    eval=config.eval_data or f"{config.data_dir}/eval.jsonl",
                )
                workflow.logger.info(
                    "skip_generate=True: train=%s eval=%s",
                    result.dataset_paths.train,
                    result.dataset_paths.eval,
                )
            else:
                if config.run_id:
                    await workflow.execute_activity(
                        update_run_status_activity,
                        args=[config.run_id, RunStatus.GENERATING.value],
                        start_to_close_timeout=timedelta(minutes=5),
                        retry_policy=_RETRY,
                    )
                result.dataset_paths = await workflow.execute_activity(
                    generate_dataset_activity,
                    DatasetConfig(
                        data_dir=config.data_dir,
                        train_size=config.train_size,
                        eval_size=config.eval_size,
                        seed=config.seed,
                        # Remote backends (k8s, kaggle, ssh, …) download training data from
                        # S3 — upload generated files immediately so the remote job finds them.
                        upload_to_storage=bool(config.remote_backend),
                        run_id=config.run_id,
                    ),
                    start_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=_RETRY,
                )

            if config.run_id:
                await workflow.execute_activity(
                    update_run_status_activity,
                    args=[config.run_id, RunStatus.TRAINING.value],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )

            result.checkpoint = await workflow.execute_activity(
                train_activity,
                TrainConfig(
                    model=config.model,
                    train_data=result.dataset_paths.train,
                    eval_data=result.dataset_paths.eval,
                    output_dir=config.output_dir,
                    epochs=config.epochs,
                    patience=config.patience,
                    warmup_ratio=config.warmup_ratio,
                    dry_run=config.dry_run,
                    remote_backend=config.remote_backend,
                    experiment_name=config.experiment_name,
                    run_id=config.run_id,
                    force_qlora=config.force_qlora,
                ),
                start_to_close_timeout=timedelta(hours=6),
                heartbeat_timeout=timedelta(minutes=10),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # ── EVAL (non-fatal) ──────────────────────────────────────────────
            if config.run_id:
                await workflow.execute_activity(
                    update_run_status_activity,
                    args=[config.run_id, RunStatus.EVALUATING.value],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )

            eval_valid_pct = 0.0
            eval_outcome_value = "failed"   # pessimistic default

            try:
                result.eval_result = await workflow.execute_activity(
                    evaluate_activity,
                    EvalConfig(
                        checkpoint=result.checkpoint.path,
                        eval_data=result.dataset_paths.eval,
                        artifact_run_id=result.checkpoint.run_id,
                        remote_backend=result.checkpoint.remote_backend,
                        output_dir=config.output_dir,
                        run_id=config.run_id,
                    ),
                    start_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )
                eval_valid_pct = result.eval_result.valid_pct
                eval_outcome_value = "succeeded" if result.eval_result.passed else "failed"
            except Exception as eval_exc:
                if is_cancelled_exception(eval_exc):
                    raise   # propagate cancellation — do not absorb it as an eval failure
                workflow.logger.warning(
                    "experiment=%s eval failed (non-fatal) — checkpoint preserved: %s",
                    config.experiment_name, eval_exc,
                    exc_info=True,
                )

            result.eval_outcome = eval_outcome_value
            result.passed = (eval_outcome_value == "succeeded")

            if config.run_id:
                await workflow.execute_activity(
                    record_eval_result_activity,
                    args=[config.run_id, eval_valid_pct, eval_outcome_value],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )

            # ── EXPORT (always runs after training succeeds) ──────────────────
            if config.run_id:
                await workflow.execute_activity(
                    update_run_status_activity,
                    args=[config.run_id, RunStatus.EXPORTING.value],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )

            result.gguf_path = await workflow.execute_activity(
                export_activity,
                ExportConfig(
                    checkpoint_path=result.checkpoint.path,
                    gguf_output=config.gguf_output,
                    run_id=result.checkpoint.run_id,
                    remote_backend="k8s",  # export always runs on k8s
                    model_id=config.model_id,
                    pipeline_run_id=config.run_id,
                    model_name=config.model_name,
                ),
                start_to_close_timeout=timedelta(hours=1),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=_NO_RETRY,
            )

            if config.model_id:
                await workflow.execute_activity(
                    save_gguf_path_activity,
                    args=[config.model_id, result.gguf_path.path],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )

            # Inference instance: created when eval succeeded, or when always_create_inference
            # is True (default) — allowing the user to start/test the model even after a
            # failed eval.
            should_create_inference = config.model_id and (
                config.always_create_inference or eval_outcome_value == "succeeded"
            )
            if should_create_inference:
                await workflow.execute_activity(
                    create_inference_activity,
                    args=[config.model_id, result.gguf_path.path, config.run_id],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )

            workflow.logger.info(
                "experiment=%s training=PASS eval=%s valid_pct=%.1f%% gguf=%s",
                config.experiment_name,
                eval_outcome_value,
                eval_valid_pct * 100,
                result.gguf_path.path,
            )

            # Training always COMPLETED — eval outcome is in eval_result field.
            if config.run_id:
                await workflow.execute_activity(
                    finalise_run_activity,
                    args=[config.run_id, True, eval_valid_pct],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )

        except Exception as exc:
            cancelled = is_cancelled_exception(exc)
            status = RunStatus.CANCELLED if cancelled else RunStatus.FAILED
            reason = "cancelled by user" if cancelled else str(exc)
            workflow.logger.warning(
                "experiment=%s %s — marking run %s as %s: %s",
                config.experiment_name,
                "cancelled" if cancelled else "failed",
                config.run_id,
                status.value,
                exc,
            )
            if config.run_id:
                await workflow.execute_activity(
                    fail_run_activity,
                    args=[config.run_id, reason, status.value],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_RETRY,
                )
            raise

        return result


# ---------------------------------------------------------------------------
# Standalone evaluate workflow (re-eval an existing run without retraining)
# ---------------------------------------------------------------------------


@dataclass
class EvaluateWorkflowConfig:
    run_id: str = ""
    remote_backend: str = ""
    remote_run_id: str = ""
    eval_data: str = "data/eval.jsonl"
    checkpoint_path: str = ""
    output_dir: str = ""


@workflow.defn
class EvaluateWorkflow:
    @workflow.run
    async def run(self, config: EvaluateWorkflowConfig) -> EvalResult:
        try:
            result = await workflow.execute_activity(
                evaluate_activity,
                EvalConfig(
                    checkpoint=config.checkpoint_path,
                    eval_data=config.eval_data,
                    artifact_run_id=config.remote_run_id,
                    remote_backend=config.remote_backend,
                    output_dir=config.output_dir,
                    run_id=config.run_id,
                ),
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=_RETRY,
            )
            if config.run_id:
                outcome = "succeeded" if result.passed else "failed"
                await workflow.execute_activity(
                    record_eval_result_activity,
                    args=[config.run_id, result.valid_pct, outcome],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )
                await workflow.execute_activity(
                    finalise_run_activity,
                    args=[config.run_id, result.passed, result.valid_pct],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )
            return result

        except Exception as exc:
            cancelled = is_cancelled_exception(exc)
            status = RunStatus.CANCELLED if cancelled else RunStatus.FAILED
            reason = "cancelled by user" if cancelled else str(exc)
            if config.run_id:
                await workflow.execute_activity(
                    fail_run_activity,
                    args=[config.run_id, reason, status.value],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_RETRY,
                )
            raise


# ---------------------------------------------------------------------------
# Standalone export workflow (download checkpoint + export GGUF)
# ---------------------------------------------------------------------------


@dataclass
class ExportWorkflowConfig:
    run_id: str = ""
    model_id: str = ""
    remote_backend: str = ""
    remote_run_id: str = ""
    checkpoint_path: str = ""
    gguf_output: str = "models/model.gguf"


@workflow.defn
class ExportWorkflow:
    @workflow.run
    async def run(self, config: ExportWorkflowConfig) -> GGUFPath:
        try:
            gguf = await workflow.execute_activity(
                export_activity,
                ExportConfig(
                    checkpoint_path=config.checkpoint_path,
                    gguf_output=config.gguf_output,
                    artifact_run_id=config.remote_run_id,
                    remote_backend="k8s",  # export always runs on k8s
                    model_id=config.model_id,
                    pipeline_run_id=config.run_id,
                ),
                start_to_close_timeout=timedelta(hours=1),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=_NO_RETRY,
            )
            if config.model_id:
                await workflow.execute_activity(
                    save_gguf_path_activity,
                    args=[config.model_id, gguf.path],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )
            if config.run_id:
                await workflow.execute_activity(
                    update_run_status_activity,
                    args=[config.run_id, "completed"],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )
            return gguf

        except Exception as exc:
            cancelled = is_cancelled_exception(exc)
            status = RunStatus.CANCELLED if cancelled else RunStatus.FAILED
            reason = "cancelled by user" if cancelled else str(exc)
            if config.run_id:
                await workflow.execute_activity(
                    fail_run_activity,
                    args=[config.run_id, reason, status.value],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_RETRY,
                )
            raise
