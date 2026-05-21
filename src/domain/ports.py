from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, TypeVar

from domain.models import (
    DatasetConfig,
    DatasetRecord,
    InferenceInstance,
    InferenceInstanceConfig,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
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


class RemoteTrainingPort(ABC):
    """Abstract interface for offloading fine-tuning to a remote compute backend.

    Implementations live in ``src/adapters/`` — never in the domain layer.

    Contract:
    - ``submit`` must start the remote job and return an opaque ``run_id``.
    - ``status`` must be non-blocking (poll, don't wait).
    - ``download`` must be called only after ``status`` returns ``"done"``; it
      fetches the checkpoint into ``dest`` and returns the local path as a string.
    """

    @abstractmethod
    def submit(self, config: RemoteTrainConfig) -> str:
        """Upload data + code and start the remote training job.

        Returns:
            An opaque ``run_id`` string used by ``status`` and ``download``.
        """

    @abstractmethod
    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        """Poll the current state of the remote job without blocking."""

    @abstractmethod
    def download(self, run_id: str, dest: Path) -> str:
        """Fetch the trained checkpoint into ``dest`` and return the local path."""

    def logs(self, run_id: str) -> str:  # noqa: ARG002
        """Return recent log output for the running job (best-effort, may be empty)."""
        return ""

    def progress(self, run_id: str) -> tuple[float, str]:  # noqa: ARG002
        """Return (completion_fraction 0.0–1.0, detail_string) for the running job.

        Override in adapters that can report structured progress.  The fraction is
        relative to the current stage (e.g. step/max_steps during training).
        """
        return 0.0, ""

    def eval(self, run_id: str, eval_data: str) -> tuple[float, bool]:  # noqa: ARG002
        """Run evaluation on the remote machine and return ``(valid_pct, passed)``.

        Raises ``NotImplementedError`` if the backend does not support remote
        evaluation (e.g. Kaggle batch kernels).  ``evaluate_activity`` catches
        this and raises an ``ApplicationError`` with a descriptive message.
        """
        raise NotImplementedError


class StorePort(ABC, Generic[TDomain, TConfig]):
    """Generic CRUD base for any domain entity store."""

    @abstractmethod
    def list(self) -> list[TDomain]:
        """Return all stored entities."""

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
    def list(self, owner_id: str | None = None) -> list[TrainingModel]:  # type: ignore[override]
        """Return all models, optionally filtered to a specific owner."""

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
    def list(self, model_id: str | None = None, owner_id: str | None = None) -> list[RunRecord]:  # type: ignore[override]
        """Return all runs, optionally filtered by model_id and/or owner_id."""

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
    def list(self, owner_id: str | None = None) -> list[DatasetRecord]:  # type: ignore[override]
        """Return all datasets, optionally filtered to a specific owner."""


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
    def list_active(self) -> list[InferenceInstance]:
        """Return instances not in SHUTDOWN or FAILED status."""


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
