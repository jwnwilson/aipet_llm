"""Unit tests for get_storage dependency in deps.py."""
from __future__ import annotations

import pytest

from interactors.api.deps import configure_storage, get_storage, clear_storage
from adapters.storage.local import LocalStorageAdapter


@pytest.fixture(autouse=True)
def _reset_storage():
    yield
    clear_storage()


def test_get_storage_raises_when_not_configured():
    clear_storage()
    with pytest.raises(RuntimeError, match="StoragePort has not been configured"):
        get_storage()


def test_get_storage_returns_configured_adapter(tmp_path):
    adapter = LocalStorageAdapter(base_dir=tmp_path)
    configure_storage(adapter)
    result = get_storage()
    assert result is adapter


def test_clear_storage_resets_to_none(tmp_path):
    adapter = LocalStorageAdapter(base_dir=tmp_path)
    configure_storage(adapter)
    clear_storage()
    with pytest.raises(RuntimeError):
        get_storage()
