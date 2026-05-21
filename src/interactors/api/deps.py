"""Shared FastAPI dependencies — all adapter and store singletons."""

from __future__ import annotations

from domain.ports import AuthPort, DatasetStorePort, InferencePort, InferenceStorePort, ModelStorePort, PodLifecyclePort, RunStorePort, StoragePort

# ---------------------------------------------------------------------------
# Inference adapter
# ---------------------------------------------------------------------------

_adapter: InferencePort | None = None


def get_adapter() -> InferencePort:
    if _adapter is None:
        raise RuntimeError("InferencePort adapter has not been configured.")
    return _adapter


def configure(adapter: InferencePort) -> None:
    global _adapter
    _adapter = adapter


def clear_adapter() -> None:
    global _adapter
    _adapter = None


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------

_model_store: ModelStorePort | None = None


def get_model_store() -> ModelStorePort:
    if _model_store is None:
        raise RuntimeError("ModelStorePort has not been configured.")
    return _model_store


def configure_model_store(store: ModelStorePort) -> None:
    global _model_store
    _model_store = store


# ---------------------------------------------------------------------------
# Run store
# ---------------------------------------------------------------------------

_run_store: RunStorePort | None = None


def get_run_store() -> RunStorePort:
    if _run_store is None:
        raise RuntimeError("RunStorePort has not been configured.")
    return _run_store


def configure_run_store(store: RunStorePort) -> None:
    global _run_store
    _run_store = store


# ---------------------------------------------------------------------------
# Auth port
# ---------------------------------------------------------------------------

_auth_port: AuthPort | None = None


def get_auth() -> AuthPort:
    if _auth_port is None:
        raise RuntimeError("AuthPort has not been configured.")
    return _auth_port


def configure_auth(port: AuthPort) -> None:
    global _auth_port
    _auth_port = port


def clear_auth() -> None:
    global _auth_port
    _auth_port = None


# ---------------------------------------------------------------------------
# Dataset store
# ---------------------------------------------------------------------------

_dataset_store: DatasetStorePort | None = None


def get_dataset_store() -> DatasetStorePort:
    if _dataset_store is None:
        raise RuntimeError("DatasetStorePort has not been configured.")
    return _dataset_store


def configure_dataset_store(store: DatasetStorePort) -> None:
    global _dataset_store
    _dataset_store = store


def clear_dataset_store() -> None:
    global _dataset_store
    _dataset_store = None


# ---------------------------------------------------------------------------
# Storage port
# ---------------------------------------------------------------------------

_storage: StoragePort | None = None


def get_storage() -> StoragePort:
    if _storage is None:
        raise RuntimeError("StoragePort has not been configured.")
    return _storage


def configure_storage(port: StoragePort) -> None:
    global _storage
    _storage = port


def clear_storage() -> None:
    global _storage
    _storage = None


# ---------------------------------------------------------------------------
# Inference store
# ---------------------------------------------------------------------------

_inference_store: InferenceStorePort | None = None


def get_inference_store() -> InferenceStorePort:
    if _inference_store is None:
        raise RuntimeError("InferenceStorePort has not been configured.")
    return _inference_store


def configure_inference_store(store: InferenceStorePort) -> None:
    global _inference_store
    _inference_store = store


def clear_inference_store() -> None:
    global _inference_store
    _inference_store = None


# ---------------------------------------------------------------------------
# Pod lifecycle adapter
# ---------------------------------------------------------------------------

_pod_adapter: PodLifecyclePort | None = None


def get_pod_adapter() -> PodLifecyclePort:
    if _pod_adapter is None:
        raise RuntimeError("PodLifecyclePort has not been configured.")
    return _pod_adapter


def configure_pod_adapter(adapter: PodLifecyclePort) -> None:
    global _pod_adapter
    _pod_adapter = adapter


def clear_pod_adapter() -> None:
    global _pod_adapter
    _pod_adapter = None
