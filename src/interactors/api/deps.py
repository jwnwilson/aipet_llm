"""Shared FastAPI dependencies — all adapter and store singletons."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends
from sqlalchemy.engine import Engine

from domain.ports import AuthPort, DatasetStorePort, InferencePort, InferenceStorePort, ModelStorePort, PodLifecyclePort, RunStorePort, StoragePort, UnitOfWorkPort

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
# Unit of Work — single entry point for all DB stores
# ---------------------------------------------------------------------------

_uow_engine: Engine | None = None


def configure_uow(engine: Engine) -> None:
    global _uow_engine
    _uow_engine = engine


def clear_uow() -> None:
    global _uow_engine
    _uow_engine = None


def get_uow() -> Generator[UnitOfWorkPort, None, None]:
    from adapters.database.uow import SQLAlchemyUnitOfWork
    if _uow_engine is None:
        raise RuntimeError("UnitOfWork has not been configured.")
    uow = SQLAlchemyUnitOfWork(_uow_engine)
    with uow.transaction():
        yield uow


# ---------------------------------------------------------------------------
# Derived store dependencies — route handlers keep their existing signatures
# ---------------------------------------------------------------------------

def get_model_store(uow: UnitOfWorkPort = Depends(get_uow)) -> ModelStorePort:
    return uow.model_store


def get_run_store(uow: UnitOfWorkPort = Depends(get_uow)) -> RunStorePort:
    return uow.run_store


def get_dataset_store(uow: UnitOfWorkPort = Depends(get_uow)) -> DatasetStorePort:
    return uow.dataset_store


def get_inference_store(uow: UnitOfWorkPort = Depends(get_uow)) -> InferenceStorePort:
    return uow.inference_store


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
