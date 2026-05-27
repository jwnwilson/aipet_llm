"""Temporal activities — one per pipeline stage, each wrapping a domain function."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from temporalio import activity

log = logging.getLogger(__name__)
from temporalio.exceptions import ApplicationError

from domain.ports import ModelStorePort, RemoteJobPort, RemoteTrainingPort, RunStorePort, StoragePort
from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE
from domain.train.config import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_PATIENCE, DEFAULT_WARMUP_RATIO


# ---------------------------------------------------------------------------
# Module-level singletons — injected by the worker (or tests)
# ---------------------------------------------------------------------------

_model_store: ModelStorePort | None = None
_run_store: RunStorePort | None = None
_storage: StoragePort | None = None


def configure_model_store(store: ModelStorePort) -> None:
    global _model_store
    _model_store = store


def configure_run_store(store: RunStorePort) -> None:
    global _run_store
    _run_store = store


def configure_storage(storage: StoragePort) -> None:
    global _storage
    _storage = storage


def _get_model_store() -> ModelStorePort:
    if _model_store is None:
        raise RuntimeError("ModelStorePort has not been configured in activities.")
    return _model_store


def _get_run_store() -> RunStorePort:
    if _run_store is None:
        raise RuntimeError("RunStorePort has not been configured in activities.")
    return _run_store


def _get_storage() -> StoragePort:
    if _storage is None:
        raise RuntimeError("StoragePort has not been configured in activities.")
    return _storage


# ---------------------------------------------------------------------------
# Config / result dataclasses (Temporal serialises these as JSON)
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    data_dir: str = "data"
    train_size: int = TRAIN_SIZE
    eval_size: int = EVAL_SIZE
    seed: int = SEED
    # Set to True when using a remote training backend (k8s, kaggle, ssh, …).
    # The activity will upload the generated files to the configured StoragePort so
    # the remote job can download them — without this the pod gets a 404 from S3.
    upload_to_storage: bool = False


@dataclass
class DatasetPaths:
    train: str = ""
    eval: str = ""


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    train_data: str = "data/train.jsonl"
    eval_data: str = "data/eval.jsonl"
    output_dir: str = DEFAULT_OUTPUT_DIR
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    dry_run: bool = False
    # Remote backend: "" or "local" → run locally; "kaggle" or "ssh" → remote.
    remote_backend: str = ""
    experiment_name: str = ""
    db_run_id: str = ""  # DB RunRecord.id for progress updates; "" = no tracking
    # None = auto-detect based on model size; True = always QLoRA; False = never QLoRA.
    force_qlora: bool | None = None


@dataclass
class CheckpointPath:
    path: str = ""            # local path; empty when checkpoint is still on the remote
    run_id: str = ""          # opaque id from adapter.submit(); non-empty for remote runs
    remote_backend: str = ""  # "kaggle", "ssh", etc.; "" means local


@dataclass
class EvalConfig:
    checkpoint: str = ""
    eval_data: str = "data/eval.jsonl"
    run_id: str = ""          # training adapter run_id (S3 prefix / kernel slug)
    remote_backend: str = ""  # non-empty → dispatch eval as EvalJobSpec to the remote GPU
    output_dir: str = ""      # local dir for downloaded eval_results.json
    db_run_id: str = ""       # DB RunRecord.id for progress updates; "" = no tracking


@dataclass
class EvalResult:
    valid_pct: float = 0.0
    passed: bool = False


@dataclass
class ExportConfig:
    checkpoint_path: str = ""
    gguf_output: str = "models/model.gguf"
    run_id: str = ""           # non-empty → download checkpoint from remote before export
    remote_backend: str = ""
    model_id: str = ""         # fallback storage key when pipeline_run_id is unset
    pipeline_run_id: str = ""  # pipeline UUID; drives storage key workflow/{id}/model.gguf
    model_name: str = ""       # human-readable name; drives gguf/{model_name}.gguf key


@dataclass
class GGUFPath:
    path: str = ""            # storage key (e.g. "gguf/{model_id}.gguf")


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


async def _heartbeat_loop(stage: str, interval: int = 30) -> None:
    """Send a liveness heartbeat every `interval` seconds while a blocking call runs."""
    while True:
        activity.heartbeat({"stage": stage})
        await asyncio.sleep(interval)


async def _poll_local_progress(db_run_id: str, output_dir: str, interval: int = 30) -> None:
    """Heartbeat loop for local training: also polls progress.json and persists it."""
    import json as _json
    progress_path = Path(output_dir) / "progress.json"
    while True:
        activity.heartbeat({"stage": "train_local"})
        if db_run_id:
            try:
                data = _json.loads(progress_path.read_text())
                step = data.get("step", 0)
                max_steps = data.get("max_steps", 1)
                frac = step / max_steps if max_steps else 0.0
                epoch = data.get("epoch", "?")
                parts = [f"epoch={epoch}"]
                for key in ("loss", "eval_loss"):
                    if key in data:
                        parts.append(f"{key}={data[key]:.4f}")
                _get_run_store().update_progress(db_run_id, frac, "  ".join(parts))
            except Exception:
                pass
        await asyncio.sleep(interval)


@activity.defn
async def generate_dataset_activity(config: DatasetConfig) -> DatasetPaths:
    from domain.train.dataset import generate

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("generate_dataset"))
    try:
        ok = await loop.run_in_executor(
            None,
            lambda: generate(
                data_dir=Path(config.data_dir),
                train_size=config.train_size,
                eval_size=config.eval_size,
                seed=config.seed,
            ),
        )
    except Exception as exc:
        raise ApplicationError(f"generate_dataset failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    if not ok:
        raise ApplicationError("Dataset generation failed: invalid examples or distribution out of bounds")

    train_path = str(Path(config.data_dir) / "train.jsonl")
    eval_path = str(Path(config.data_dir) / "eval.jsonl")

    if config.upload_to_storage:
        # Remote backends (K8s, Kaggle, SSH, …) fetch training data from S3.
        # Upload the generated files now so the remote job can call storage.download().
        storage = _get_storage()
        await loop.run_in_executor(
            None,
            lambda: (
                storage.upload(Path(train_path), train_path),
                storage.upload(Path(eval_path), eval_path),
            ),
        )
        log.info("Dataset uploaded to storage: train=%s eval=%s", train_path, eval_path)

    return DatasetPaths(train=train_path, eval=eval_path)


def _make_remote_adapter(backend: str) -> RemoteJobPort:
    # Storage is only needed for S3-backed backends; fetch lazily to avoid
    # failing when storage is not configured (e.g. Kaggle tests).
    if backend == "k8s":
        from adapters.compute.k8s.adapter import K8sTrainingAdapter
        return K8sTrainingAdapter(storage=_get_storage())
    if backend == "kaggle":
        from adapters.compute.kaggle import KaggleTrainingAdapter
        return KaggleTrainingAdapter()          # Kaggle uses its own file-based staging
    if backend == "ssh":
        from adapters.compute.ssh import SshTrainingAdapter
        return SshTrainingAdapter()             # SSH has no StoragePort injection yet
    if backend == "colab":
        from adapters.compute.colab.adapter import ColabTrainingAdapter
        return ColabTrainingAdapter()           # Colab has no StoragePort injection yet
    if backend == "runpod":
        from adapters.compute.runpod import RunPodTrainingAdapter
        return RunPodTrainingAdapter(storage=_get_storage())
    raise ApplicationError(f"Unknown remote_backend: {backend!r}")


@activity.defn
async def train_activity(config: TrainConfig) -> CheckpointPath:
    backend = config.remote_backend or "local"

    if backend == "local":
        return await _train_local(config)

    adapter = _make_remote_adapter(backend)
    return await _train_remote(config, adapter)


async def _train_local(config: TrainConfig) -> CheckpointPath:
    from domain.train.trainer import train

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(
        _poll_local_progress(config.db_run_id, config.output_dir)
    )
    try:
        await loop.run_in_executor(
            None,
            lambda: train(
                model=config.model,
                train_data=config.train_data,
                eval_data=config.eval_data,
                output_dir=config.output_dir,
                epochs=config.epochs,
                patience=config.patience,
                warmup_ratio=config.warmup_ratio,
                dry_run=config.dry_run,
                force_qlora=config.force_qlora,
            ),
        )
    except Exception as exc:
        raise ApplicationError(f"train failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    return CheckpointPath(path=config.output_dir)


# Backends that stage training data from the local filesystem before uploading it
# to their own compute environment.  These need train/eval files materialised
# locally even when skip_generate=True provides an S3 key as train_data.
#
# S3-backed backends (k8s, runpod) pass the S3 key directly to the
# remote job; the pod downloads from S3 itself.  Replacing the key with a local
# path would point the pod at a key that doesn't exist → DO NOT download for them.
# Note: Kaggle is intentionally excluded here — it accesses training data from S3
# directly (enable_internet=True) rather than requiring local staging.
_FILE_BASED_BACKENDS = frozenset({"colab", "ssh"})


async def _resolve_training_data(
    config: TrainConfig, loop: asyncio.AbstractEventLoop
) -> tuple[str, str]:
    """Return *(train_key, eval_key)* ready for use in ``TrainJobSpec``.

    For **file-based backends** (Kaggle, Colab, SSH) the adapter stages data
    from the local filesystem.  When ``skip_generate=True`` the workflow
    forwards an S3 storage key (e.g. ``datasets/<uuid>.jsonl``) as
    ``train_data`` — the file does not exist locally.  This function downloads
    it to ``data/train.jsonl`` / ``data/eval.jsonl`` and returns the local paths.

    For **S3-backed backends** (K8s, RunPod, VastAI) the remote job downloads
    from S3 directly; the original S3 keys are returned unchanged so the pod
    can find them (replacing keys with local paths would break S3 downloads).

    If the paths already exist locally this is a no-op for file-based backends too.
    """
    train_key = config.train_data
    # Derive eval key: use explicit config value or the sibling eval.jsonl.
    eval_key = config.eval_data or str(Path(train_key).parent / "eval.jsonl")

    # S3-backed backends: pass keys straight through — no local download needed.
    if config.remote_backend not in _FILE_BASED_BACKENDS:
        return train_key, eval_key

    def _is_local(key: str) -> bool:
        p = Path(key)
        return (p if p.is_absolute() else Path.cwd() / p).exists()

    if _is_local(train_key) and _is_local(eval_key):
        return train_key, eval_key

    activity.logger.info(
        "Training data not found locally (train=%r eval=%r cwd=%s); "
        "downloading from storage for local staging.",
        train_key, eval_key, Path.cwd(),
    )
    storage = _get_storage()
    dest = Path("data")
    dest.mkdir(parents=True, exist_ok=True)
    local_train = dest / "train.jsonl"
    local_eval = dest / "eval.jsonl"

    if not _is_local(train_key):
        await loop.run_in_executor(None, lambda: storage.download(train_key, local_train))
        activity.logger.info("Downloaded train data → %s", local_train)

    if not _is_local(eval_key):
        await loop.run_in_executor(None, lambda: storage.download(eval_key, local_eval))
        activity.logger.info("Downloaded eval data → %s", local_eval)

    return str(local_train), str(local_eval)


async def _train_remote(config: TrainConfig, adapter: RemoteJobPort) -> CheckpointPath:
    from domain.models import TrainJobSpec

    loop = asyncio.get_event_loop()

    # Capture original keys before resolution — backends that access S3 directly
    # (Kaggle, K8s, RunPod) use these to configure the remote worker without
    # staging data on the local filesystem.
    original_train_key = config.train_data
    original_eval_key = config.eval_data

    # If train/eval paths are S3 keys, download them locally for file-based
    # backends (Colab, SSH) that stage data into their compute environment.
    train_data, eval_data = await _resolve_training_data(config, loop)

    remote_config = TrainJobSpec(
        model=config.model,
        train_data=train_data,
        eval_data=eval_data,
        epochs=config.epochs,
        patience=config.patience,
        warmup_ratio=config.warmup_ratio,
        experiment_name=config.experiment_name or "llm-api",
        # Thread the DB run-record UUID so K8s can use it as the S3 key prefix,
        # ensuring the training upload and export download use the same path.
        db_run_id=config.db_run_id,
        # Original S3 keys for backends that access storage directly (e.g. Kaggle).
        train_s3_key=original_train_key,
        eval_s3_key=original_eval_key,
    )

    # Resume from a prior attempt if the remote job was already submitted.
    info = activity.info()
    prior_run_id: str | None = None
    if info.heartbeat_details:
        prev = info.heartbeat_details[0]
        if isinstance(prev, dict) and prev.get("run_id"):
            prior_run_id = prev["run_id"]

    if prior_run_id:
        activity.logger.info(
            "Resuming poll for existing remote job: adapter=%s run_id=%s",
            type(adapter).__name__, prior_run_id,
        )
        run_id = prior_run_id
    else:
        # Run submit in an executor — it calls subprocess + time.sleep (blocks event loop).
        heartbeat_task = asyncio.ensure_future(_heartbeat_loop("train_submit"))
        try:
            run_id = await loop.run_in_executor(None, lambda: adapter.submit(remote_config))
        except Exception as exc:
            raise ApplicationError(f"Remote submit failed: {exc}") from exc
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

        activity.logger.info("Remote job submitted: adapter=%s run_id=%s", type(adapter).__name__, run_id)

    started_at = time.time()

    # Background heartbeat keeps the activity alive while executor calls block.
    # status() + logs() can take >2 min on slow VastAI/S3 paths; without this
    # the heartbeat_timeout fires before the inline heartbeat arrives.
    poll_heartbeat = asyncio.ensure_future(_heartbeat_loop("train_poll", interval=30))
    try:
        while True:
            # status/logs/progress all run subprocess — keep them off the event loop.
            try:
                status = await loop.run_in_executor(None, lambda: adapter.status(run_id))
            except Exception as exc:
                raise ApplicationError(f"Remote status check failed: {exc}") from exc

            elapsed_s = int(time.time() - started_at)
            logs = await loop.run_in_executor(None, lambda: adapter.logs(run_id))

            activity.logger.info(
                "Remote status: adapter=%s run_id=%s status=%s elapsed=%ds",
                type(adapter).__name__, run_id, status, elapsed_s,
            )
            log.info(
                "Remote status: adapter=%s run_id=%s status=%s elapsed=%ds",
                type(adapter).__name__, run_id, status, elapsed_s,
            )
            if logs:
                activity.logger.info("Instance output:\n%s", logs)
                log.info("Instance output (run_id=%s):\n%s", run_id, logs)

            if logs and config.db_run_id:
                try:
                    log_path = Path(f"data/workflow/{config.db_run_id}/logs.txt")
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(logs)
                except Exception:
                    log.warning("Failed to persist training logs for run %s", config.db_run_id)

            # Detailed heartbeat after each poll (background loop covers gaps between polls).
            # run_id is included so a restarted worker can resume polling without resubmitting.
            activity.heartbeat({"status": status, "elapsed_s": elapsed_s, "logs": logs, "run_id": run_id})

            if config.db_run_id:
                try:
                    frac, detail = await loop.run_in_executor(None, lambda: adapter.progress(run_id))
                    if frac > 0:
                        _get_run_store().update_progress(config.db_run_id, frac, detail)
                except Exception:
                    pass

            if status == "done":
                # Defer download — eval may run on the remote, so we avoid pulling
                # gigabytes of checkpoint data until we know the model actually passes.
                return CheckpointPath(
                    run_id=run_id,
                    remote_backend=config.remote_backend,
                )

            if status == "failed":
                tail = f"\n--- pod logs ---\n{logs}" if logs else " (no logs captured)"
                raise ApplicationError(
                    f"Remote training failed (adapter={type(adapter).__name__}, run_id={run_id}){tail}"
                )

            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                activity.logger.warning(
                    "train_activity cancelled while polling (adapter=%s, run_id=%s, elapsed=%ds)",
                    type(adapter).__name__,
                    run_id,
                    int(time.time() - started_at),
                )
                raise
    finally:
        poll_heartbeat.cancel()
        await asyncio.gather(poll_heartbeat, return_exceptions=True)


@activity.defn
async def evaluate_activity(config: EvalConfig) -> EvalResult:
    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("evaluate"))
    try:
        if config.remote_backend:
            # Dispatch eval to the remote GPU that ran training.
            # The EvalJobSpec downloads checkpoint + eval data on the remote machine;
            # the Temporal worker only polls status and retrieves the tiny eval_results.json.
            result = await _evaluate_via_remote_job(config, loop)
        else:
            result = await _evaluate_local(config, loop)
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(f"evaluate failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    activity.logger.info(
        "Eval complete: valid_pct=%.1f%% passed=%s",
        result.valid_pct * 100,
        result.passed,
    )
    return result


async def _evaluate_via_remote_job(config: EvalConfig, loop: asyncio.AbstractEventLoop) -> EvalResult:
    """Submit an EvalJobSpec to the remote backend and poll until done.

    Mirrors the _train_remote polling loop so heartbeats are sent throughout.
    """
    import json as _json
    from domain.models import EvalJobSpec

    adapter = _make_remote_adapter(config.remote_backend)
    spec = EvalJobSpec(
        experiment_name=f"eval-{config.db_run_id or 'standalone'}",
        training_artifact_ref=config.run_id,
        eval_data=config.eval_data,
        # db_run_id lets the Kaggle eval kernel locate the checkpoint in S3 at
        # workflow/{db_run_id}/checkpoint/ rather than pulling from kernel output.
        db_run_id=config.db_run_id,
    )
    eval_run_id = await loop.run_in_executor(None, lambda: adapter.submit(spec))
    activity.logger.info(
        "Remote eval submitted: backend=%s eval_run_id=%s", config.remote_backend, eval_run_id
    )

    poll_hb = asyncio.ensure_future(_heartbeat_loop("eval_poll", interval=30))
    try:
        while True:
            status = await loop.run_in_executor(None, lambda: adapter.status(eval_run_id))
            logs = await loop.run_in_executor(None, lambda: adapter.logs(eval_run_id))
            activity.heartbeat({"status": status, "eval_run_id": eval_run_id})
            if logs:
                activity.logger.info("Remote eval output:\n%s", logs)
            if status == "done":
                break
            if status == "failed":
                raise ApplicationError(
                    f"Remote eval failed (backend={config.remote_backend}, "
                    f"eval_run_id={eval_run_id})\n{logs}"
                )
            await asyncio.sleep(30)
    finally:
        poll_hb.cancel()
        await asyncio.gather(poll_hb, return_exceptions=True)

    dest = Path(config.output_dir or f"models/eval_tmp/{eval_run_id.replace('/', '_')}")
    try:
        result_path = await loop.run_in_executor(None, lambda: adapter.download(eval_run_id, dest))
        data = _json.loads(Path(result_path).read_text())
        return EvalResult(valid_pct=data["valid_pct"], passed=data["passed"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ApplicationError(
            f"Remote eval result malformed "
            f"(backend={config.remote_backend}, eval_run_id={eval_run_id}): {exc}"
        ) from exc
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(
            f"Remote eval result download failed "
            f"(backend={config.remote_backend}, eval_run_id={eval_run_id}): {exc}"
        ) from exc


def _normalise_report_keys(obj: object) -> object:
    """Recursively rename the JSON key 'pass' → 'passed' (pass is a Python keyword)."""
    if isinstance(obj, dict):
        return {("passed" if k == "pass" else k): _normalise_report_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalise_report_keys(v) for v in obj]
    return obj


async def _evaluate_local(config: EvalConfig, loop: asyncio.AbstractEventLoop) -> EvalResult:
    from domain.train.evaluate import evaluate, infer_hf, load_hf_pipeline

    pipe = load_hf_pipeline(config.checkpoint)
    infer_fn_raw = lambda prompt: infer_hf(pipe, prompt)  # noqa: E731

    exit_code, valid_pct = await loop.run_in_executor(
        None, lambda: evaluate(Path(config.eval_data), infer_fn_raw)
    )

    if config.db_run_id:
        from adapters.prompt import build_prompt, parse_response
        from domain.train.quality_report import run_quality_report

        infer_fn_rich = lambda req: parse_response(infer_fn_raw(build_prompt(req)))  # noqa: E731
        report = await loop.run_in_executor(
            None, lambda: run_quality_report(infer_fn_rich)
        )
        report = _normalise_report_keys(report)
        report_path = Path(f"data/workflow/{config.db_run_id}/quality_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report))
        activity.logger.info("Quality report saved: %s", report_path)

    return EvalResult(valid_pct=valid_pct, passed=exit_code == 0)


@activity.defn
async def export_activity(config: ExportConfig) -> GGUFPath:
    # Determine the destination GGUF storage key upfront — shared by all paths.
    if config.model_name:
        gguf_key = f"gguf/{config.model_name}.gguf"
    elif config.pipeline_run_id:
        gguf_key = f"workflow/{config.pipeline_run_id}/model.gguf"
    elif config.model_id:
        gguf_key = f"gguf/{config.model_id}.gguf"
    else:
        gguf_key = config.gguf_output

    # ── K8s path: dispatch a dedicated export Job ─────────────────────────
    # The Temporal worker image does NOT have llama.cpp installed.  When the
    # backend is "k8s" we submit an ExportJobSpec so the conversion runs inside
    # the export Docker image (which has llama.cpp pre-built).
    # The Job downloads the checkpoint from S3, converts it, and uploads the
    # GGUF back to S3 — the worker only needs to submit + poll.
    if config.remote_backend == "k8s":
        from domain.models import ExportJobSpec
        db_run_id = config.pipeline_run_id
        if not db_run_id:
            raise ApplicationError(
                "export_activity: pipeline_run_id is required for K8s export "
                "(it is the DB run_id used as the S3 key prefix)."
            )
        spec = ExportJobSpec(
            experiment_name=db_run_id,
            checkpoint_s3_prefix=f"workflow/{db_run_id}/checkpoint/",
            gguf_s3_key=gguf_key,
        )
        adapter = _make_remote_adapter("k8s")
        loop = asyncio.get_event_loop()
        heartbeat_task = asyncio.ensure_future(_heartbeat_loop("export"))
        try:
            export_run_id = await loop.run_in_executor(None, lambda: adapter.submit(spec))
            activity.logger.info(
                "K8s export Job submitted: %s (db_run_id=%s, gguf_key=%s)",
                export_run_id,
                db_run_id,
                gguf_key,
            )
            while True:
                status = await loop.run_in_executor(
                    None, lambda: adapter.status(export_run_id)
                )
                if status == "done":
                    break
                if status == "failed":
                    logs = adapter.logs(export_run_id)
                    raise ApplicationError(
                        f"K8s export Job {export_run_id} failed.\nLogs:\n{logs}"
                    )
                await asyncio.sleep(30)
        except ApplicationError:
            raise
        except Exception as exc:
            raise ApplicationError(f"K8s export failed: {exc}") from exc
        finally:
            heartbeat_task.cancel()
        return GGUFPath(path=gguf_key)

    # ── Local / other-remote path: download checkpoint then export inline ──
    # Non-k8s backends (kaggle, ssh, colab, runpod) and local runs
    # still download the checkpoint to the worker and export inline.
    from domain.train.export import export as export_gguf

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("export"))
    try:
        if config.remote_backend:
            adapter = _make_remote_adapter(config.remote_backend)
            dest = Path(config.gguf_output).parent / "checkpoint"
            try:
                checkpoint_path = await loop.run_in_executor(
                    None, lambda: adapter.download(config.run_id, dest)
                )
            except Exception as exc:
                raise ApplicationError(f"Remote download failed: {exc}") from exc
        else:
            checkpoint_path = config.checkpoint_path

        local_gguf = Path(config.gguf_output)
        await loop.run_in_executor(
            None,
            lambda: export_gguf(checkpoint=Path(checkpoint_path), output=local_gguf),
        )

        # Upload to storage so the API can retrieve it by key.
        from adapters.storage import upload_model
        storage = _get_storage()
        gguf_key = upload_model(storage, local_gguf, gguf_key)

    except SystemExit as exc:
        raise ApplicationError(f"export failed: llama.cpp setup issue (exit {exc.code})") from exc
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(f"export failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    return GGUFPath(path=gguf_key)


@activity.defn
async def finalise_run_activity(run_id: str, passed: bool, valid_pct: float) -> None:
    """Mark the run as completed or failed.

    ``passed`` reflects whether *training* succeeded (not eval) — the workflow
    always calls this with ``passed=True`` after a successful train.  Eval outcome
    is recorded separately via ``record_eval_result_activity``.
    The ``valid_pct`` parameter is kept for backward-compat but is no longer
    persisted here (``record_eval_result_activity`` owns that).
    """
    from domain.models import RunStatus

    store = _get_run_store()
    store.update_status(run_id, RunStatus.COMPLETED if passed else RunStatus.FAILED)
    activity.logger.info(
        "Run %s finalised: status=%s",
        run_id,
        RunStatus.COMPLETED.value if passed else RunStatus.FAILED.value,
    )


@activity.defn
async def record_eval_result_activity(
    run_id: str, valid_pct: float, outcome_value: str
) -> None:
    """Persist the eval score and SUCCEEDED/FAILED outcome atomically.

    Called by the workflow after evaluate_activity finishes (or raises).
    Keeping this as a separate activity means eval failure never blocks
    the run from being marked COMPLETED.
    """
    from domain.models import EvalOutcome

    store = _get_run_store()
    store.update_eval_result(run_id, valid_pct, EvalOutcome(outcome_value))
    activity.logger.info(
        "Eval result recorded: run=%s pct=%.1f%% outcome=%s",
        run_id, valid_pct * 100, outcome_value,
    )


@activity.defn
async def update_run_status_activity(run_id: str, status_value: str) -> None:
    """Set run status without touching eval_valid_pct (used by export-only workflows)."""
    from domain.models import RunStatus
    store = _get_run_store()
    store.update_status(run_id, RunStatus(status_value))


@activity.defn
async def fail_run_activity(run_id: str, reason: str, status_value: str = "failed") -> None:
    """Mark a run as failed or cancelled, persisting the reason in progress_detail.

    Called by the workflow's exception handlers so that any activity failure or
    Temporal cancellation is reflected in the run record.
    """
    from domain.models import RunStatus
    store = _get_run_store()
    store.fail_run(run_id, reason, RunStatus(status_value))
    activity.logger.info(
        "Run %s marked %s: %s",
        run_id,
        status_value,
        reason,
    )


@activity.defn
async def save_gguf_path_activity(model_id: str, gguf_path: str) -> None:
    """Persist the storage key of the exported GGUF back to the model record."""
    store = _get_model_store()
    model = store.get(model_id)
    if model is None:
        activity.logger.warning("save_gguf_path: model %s not found — skipping", model_id)
        return

    from domain.models import TrainingModelConfig
    config = TrainingModelConfig(
        name=model.name,
        description=model.description,
        base_model=model.base_model,
        train_data=model.train_data,
        eval_data=model.eval_data,
        epochs=model.epochs,
        patience=model.patience,
        warmup_ratio=model.warmup_ratio,
        remote_backend=model.remote_backend,
        skip_generate=model.skip_generate,
        gguf_path=gguf_path,
        is_active=model.is_active,
    )
    store.update(model_id, config)
    activity.logger.info("Saved gguf_path=%s for model %s", gguf_path, model_id)


@activity.defn
async def create_inference_activity(model_id: str) -> str:
    """Create an InferenceInstance record for a successfully exported model.
    Returns the new instance id."""
    from domain.models import InferenceInstanceConfig
    from interactors.api import deps
    store = deps.get_inference_store()
    config = InferenceInstanceConfig(model_id=model_id)
    instance = store.create(config)
    return instance.id
