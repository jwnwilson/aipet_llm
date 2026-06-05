from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, TypeVar

from domain.models import (
    DatasetConfig,
    DatasetRecord,
    EvalOutcome,
    InferenceInstance,
    InferenceInstanceConfig,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    RemoteJobSpec,
    RemoteTrainConfig,
    RunConfig,
    RunRecord,
    RunStatus,
    TrainingModel,
    TrainingModelConfig,
    UserContext,
)

TDomain = TypeVar("TDomain")
TConfig = TypeVar("TConfig")


class StoragePort(ABC):
    """Abstract interface for storing and retrieving model artifact files (GGUFs, checkpoints).

    Keys are relative paths such as ``workflow/{run_id}/model.gguf``.  Backends map
    these to their own namespace (local filesystem prefix, S3 key prefix, etc.).
    """

    @abstractmethod
    def upload(self, local_path: Path, key: str) -> None:
        """Copy a local file into storage under ``key``."""

    @abstractmethod
    def download(self, key: str, dest: Path) -> None:
        """Fetch the artifact at ``key`` to ``dest`` (creates parent dirs).

        Must be a no-op when the source and destination resolve to the same path
        (i.e. local storage where the file is already in place).
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if ``key`` exists in storage."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` from storage (silent no-op if already absent)."""

    def write_bytes(self, key: str, content: bytes) -> None:
        """Write raw bytes to ``key`` (creates or overwrites).

        Override in adapters that support arbitrary key writes.
        Raises ``NotImplementedError`` on adapters that do not.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement write_bytes")

    def read_text(self, key: str, *, encoding: str = "utf-8") -> str:
        """Read ``key`` and return decoded text; return \"\" if the key is absent.

        Override in adapters that support object reads.
        Raises ``NotImplementedError`` on adapters that do not.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement read_text")

    def read_bytes_from(self, key: str, offset: int = 0) -> bytes:
        """Read bytes from ``key`` starting at ``offset``; return b\"\" if absent or offset >= size.

        Used for incremental log streaming — callers track ``offset`` themselves.
        Override in adapters that support range reads.
        Raises ``NotImplementedError`` on adapters that do not.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement read_bytes_from")

    def download_directory(self, prefix: str, dest) -> None:
        """Download all objects whose key starts with ``prefix`` into ``dest``.

        ``prefix`` should end with ``/``. Relative paths within the prefix are
        preserved under ``dest``. Override in adapters that support prefix listing.
        Raises ``NotImplementedError`` on adapters that do not.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement download_directory")

    def delete_directory(self, prefix: str) -> None:
        """Delete all objects whose key starts with ``prefix``.

        ``prefix`` should end with ``/`` (e.g. ``workflow/{run_id}/checkpoint/``).
        Silent no-op if no objects match. Override in adapters that support prefix
        deletion. Raises ``NotImplementedError`` on adapters that do not.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement delete_directory")

    def upload_directory(self, local_dir: Path, prefix: str) -> None:
        """Upload all files under ``local_dir`` to storage, keyed as ``{prefix}/{relative_path}``.

        ``prefix`` must NOT end with ``/``.  Symmetric counterpart to
        ``download_directory``: calling ``upload_directory(d, p)`` followed by
        ``download_directory(p + '/', dest)`` reproduces ``d`` under ``dest``.

        The default implementation iterates files and delegates to ``self.upload()``,
        which every concrete adapter must implement.  Adapters may override for
        efficiency (e.g. parallel S3 multipart uploads).

        Raises:
            ValueError: if ``prefix`` ends with ``/``.
            RuntimeError: if one or more files fail to upload (all files are
                attempted; failures are collected and reported together).
        """
        if prefix.endswith("/"):
            raise ValueError(f"prefix must not end with '/': {prefix!r}")

        failed: list[tuple[Path, Exception]] = []
        for file_path in sorted(Path(local_dir).rglob("*")):
            if file_path.is_file():
                relative = file_path.relative_to(local_dir)
                try:
                    self.upload(file_path, f"{prefix}/{relative}")
                except Exception as exc:  # noqa: BLE001
                    failed.append((file_path, exc))

        if failed:
            details = "; ".join(f"{p.name}: {e}" for p, e in failed)
            raise RuntimeError(
                f"upload_directory incomplete — {len(failed)} file(s) failed: {details}"
            )


class InferencePort(ABC):
    """Abstract interface the domain layer expects from any LLM inference backend.

    Contract:
    - ``infer`` must always return a valid ``InferenceResponse``.
    - ``infer`` must never raise on recoverable LLM errors; instead return
      ``InferenceResponse(action=Action.IDLE)`` so the pet remains in a safe,
      neutral state while the problem is handled upstream.
    """

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference for the given request and return a structured response.

        Args:
            request: The scene and pet-stat context to reason over.

        Returns:
            A valid ``InferenceResponse``.  On recoverable errors the
            implementation must return ``InferenceResponse(action=Action.IDLE)``
            rather than raising an exception.
        """


@dataclass
class SubmitRetryConfig:
    """Retry constraints for adapter.submit(). Each backend overrides submit_retry_config()."""

    max_retries: int = 3
    base_delay_s: float = 5.0
    retryable_errors: tuple[str, ...] = field(default_factory=lambda: (
        "rate limit",
        "temporarily unavailable",
        "timeout",
        "too many requests",
        "service unavailable",
    ))


class RemoteJobPort(ABC):
    """Abstract interface for dispatching compute jobs (train, eval, …) to remote backends.

    Implementations live in ``src/adapters/`` — never in the domain layer.

    Contract:
    - ``submit`` must start the remote job and return an opaque ``run_id``.
    - ``status`` must be non-blocking (poll, don't wait).
    - ``download`` must be called only after ``status`` returns ``"done"``; it
      fetches job artifacts into ``dest`` and returns the local path as a string.
      Train jobs return the checkpoint directory path.
      Eval jobs return the eval_results.json file path.
    """

    @abstractmethod
    def submit(self, spec: RemoteJobSpec) -> str:
        """Stage resources, launch job, return opaque run_id.

        Args:
            spec: Either a ``TrainJobSpec`` or ``EvalJobSpec``, selected by ``job_type``.

        Returns:
            An opaque ``run_id`` string used by ``status``, ``logs``, and ``download``.
        """

    @abstractmethod
    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        """Poll the current state of the remote job without blocking."""

    @abstractmethod
    def download(self, run_id: str, dest: Path) -> str:
        """Fetch job artifacts into ``dest`` and return the local path.

        - Train jobs: returns the checkpoint directory path.
        - Eval jobs: returns the eval_results.json file path.
        """

    def logs(self, run_id: str) -> str:  # noqa: ARG002
        """Return recent log output for the running job (best-effort, may be empty)."""
        return ""

    def progress(self, run_id: str) -> tuple[float, str]:  # noqa: ARG002
        """Return (completion_fraction 0.0–1.0, detail_string) for the running job.

        Override in adapters that can report structured progress.  The fraction is
        relative to the current stage (e.g. step/max_steps during training).
        """
        return 0.0, ""

    def submit_retry_config(self) -> SubmitRetryConfig:
        """Return retry constraints for submit(). Override per backend to customise."""
        return SubmitRetryConfig()


# Backward-compat alias — existing code using RemoteTrainingPort continues to work.
RemoteTrainingPort = RemoteJobPort


class StorePort(ABC, Generic[TDomain, TConfig]):
    """Generic CRUD base for any domain entity store."""

    @abstractmethod
    def list(self, offset: int = 0, limit: int = 50) -> list[TDomain]:
        """Return stored entities with optional offset/limit for pagination."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored entities."""

    @abstractmethod
    def get(self, id: str) -> TDomain | None:
        """Return the entity with the given id, or None if not found."""

    @abstractmethod
    def create(self, config: TConfig) -> TDomain:
        """Persist a new entity and return it with id and timestamps."""

    @abstractmethod
    def update(self, id: str, config: TConfig) -> TDomain | None:
        """Update an existing entity; return updated entity or None if not found."""

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete an entity by id; return True if deleted, False if not found."""


class ModelStorePort(StorePort["TrainingModel", "TrainingModelConfig"]):
    """Abstract interface for persisting training model configurations."""

    @abstractmethod
    def list(self, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[TrainingModel]:  # type: ignore[override]
        """Return models with optional owner filter and pagination."""

    @abstractmethod
    def count(self, owner_id: str | None = None) -> int:  # type: ignore[override]
        """Return total model count, optionally filtered by owner."""

    @abstractmethod
    def activate(self, id: str) -> TrainingModel | None:
        """Set ``is_active=True`` for this model, ``False`` for all others.

        Returns the updated model, or ``None`` if ``id`` is not found.
        """

    @abstractmethod
    def active(self) -> TrainingModel | None:
        """Return the currently active model, or ``None`` if none is set."""


class RunStorePort(StorePort["RunRecord", "RunConfig"]):
    """Abstract interface for persisting training run records."""

    @abstractmethod
    def list(self, model_id: str | None = None, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[RunRecord]:  # type: ignore[override]
        """Return runs with optional filters and pagination."""

    @abstractmethod
    def count(self, model_id: str | None = None, owner_id: str | None = None) -> int:  # type: ignore[override]
        """Return total run count matching the given filters."""

    @abstractmethod
    def update_status(self, run_id: str, status: RunStatus) -> RunRecord | None:
        """Set the run status; return updated record or None if not found."""

    @abstractmethod
    def update_eval(self, run_id: str, valid_pct: float) -> RunRecord | None:
        """Persist the eval result; return updated record or None if not found."""

    @abstractmethod
    def update_progress(self, run_id: str, progress: float, detail: str = "") -> RunRecord | None:
        """Persist training/evaluation progress fraction and detail string."""

    @abstractmethod
    def fail_run(
        self,
        run_id: str,
        reason: str,
        status: RunStatus = RunStatus.FAILED,
    ) -> RunRecord | None:
        """Mark a run as failed or cancelled, persisting the reason in progress_detail.

        Returns the updated record, or None if run_id is not found.
        """

    @abstractmethod
    def update_eval_result(
        self, run_id: str, valid_pct: float, outcome: EvalOutcome
    ) -> RunRecord | None:
        """Persist eval score and pass/fail outcome atomically.

        Returns the updated record, or None if run_id is not found.
        """


class AuthPort(ABC):
    """Abstract interface for validating bearer tokens."""

    @abstractmethod
    def authenticate(self, token: str) -> UserContext | None:
        """Validate the JWT and return a UserContext, or None if invalid/expired."""


class DatasetStorePort(StorePort["DatasetRecord", "DatasetConfig"]):
    """Abstract interface for persisting dataset metadata records.

    The actual file content lives in a ``StoragePort`` under ``key``.
    This store tracks the metadata: name, type, storage key, timestamps.
    """

    @abstractmethod
    def list(self, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[DatasetRecord]:  # type: ignore[override]
        """Return datasets with optional owner filter and pagination."""

    @abstractmethod
    def count(self, owner_id: str | None = None) -> int:  # type: ignore[override]
        """Return total dataset count, optionally filtered by owner."""


class InferenceStorePort(StorePort["InferenceInstance", "InferenceInstanceConfig"]):
    """Abstract interface for persisting inference instance records."""

    @abstractmethod
    def update_status(self, id: str, status: InferenceStatus) -> InferenceInstance | None:
        """Set the instance status; return updated record or None if not found."""

    @abstractmethod
    def update_pod(self, id: str, pod_name: str, pod_namespace: str) -> InferenceInstance | None:
        """Set the pod name and namespace; return updated record or None if not found."""

    @abstractmethod
    def update_last_used(self, id: str) -> InferenceInstance | None:
        """Set last_used_at to now; return updated record or None if not found."""

    @abstractmethod
    def list(self, model_id: str | None = None, run_id: str | None = None, offset: int = 0, limit: int = 50) -> list[InferenceInstance]:  # type: ignore[override]
        """Return inference instances with optional model/run filter and pagination."""

    @abstractmethod
    def count(self, model_id: str | None = None, run_id: str | None = None) -> int:  # type: ignore[override]
        """Return total inference instance count, optionally filtered by model or run."""

    @abstractmethod
    def list_active(self) -> list[InferenceInstance]:
        """Return instances not in SHUTDOWN or FAILED status."""

    @abstractmethod
    def delete_by_model(self, model_id: str) -> list[InferenceInstance]:
        """Delete all inference instances for model_id. Returns the deleted instances."""


class PodLifecyclePort(ABC):
    """Abstract interface for managing inference pod lifecycle."""

    @abstractmethod
    def create_pod(self, pod_name: str, model_id: str, model_path: str, namespace: str = "default") -> str:
        """Create the inference pod and paired Service. Return pod_name."""

    @abstractmethod
    def pod_status(self, pod_name: str, namespace: str = "default") -> Literal["pending", "running", "failed", "unknown"]:
        """Return the current pod phase without blocking."""

    @abstractmethod
    def delete_pod(self, pod_name: str, namespace: str = "default") -> None:
        """Delete pod and its Service. No-op if already gone."""

    @abstractmethod
    def pod_service_url(self, pod_name: str, namespace: str = "default") -> str:
        """Return the ClusterIP HTTP URL for routing inference requests to this pod."""
