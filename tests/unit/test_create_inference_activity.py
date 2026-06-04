"""Unit tests for create_inference_activity."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


def _mock_uow(inference_store=None):
    """Mock UoW whose transaction() context manager yields itself."""
    mock = MagicMock()
    if inference_store is not None:
        mock.inference_store = inference_store

    @contextmanager
    def fake_transaction():
        yield mock

    mock.transaction = fake_transaction
    return mock


class TestCreateInferenceActivity:
    """Tests for create_inference_activity."""

    @pytest.mark.asyncio
    async def test_creates_instance_with_correct_model_id(self):
        mock_instance = MagicMock()
        mock_instance.id = "inst-abc123"
        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        import interactors.temporal.activities as acts
        original = acts._create_uow
        acts._create_uow = lambda: _mock_uow(inference_store=mock_store)
        try:
            from interactors.temporal.activities import create_inference_activity
            result = await create_inference_activity("model-42")
        finally:
            acts._create_uow = original

        call_args = mock_store.create.call_args[0][0]
        assert call_args.model_id == "model-42"

    @pytest.mark.asyncio
    async def test_creates_instance_with_model_path(self):
        mock_instance = MagicMock()
        mock_instance.id = "inst-def456"
        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        import interactors.temporal.activities as acts
        original = acts._create_uow
        acts._create_uow = lambda: _mock_uow(inference_store=mock_store)
        try:
            from interactors.temporal.activities import create_inference_activity
            result = await create_inference_activity("model-42", "workflow/abc/model.gguf")
        finally:
            acts._create_uow = original

        call_args = mock_store.create.call_args[0][0]
        assert call_args.model_id == "model-42"
        assert call_args.model_path == "workflow/abc/model.gguf"

    @pytest.mark.asyncio
    async def test_creates_instance_with_empty_model_path_by_default(self):
        mock_instance = MagicMock()
        mock_instance.id = "inst-ghi789"
        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        import interactors.temporal.activities as acts
        original = acts._create_uow
        acts._create_uow = lambda: _mock_uow(inference_store=mock_store)
        try:
            from interactors.temporal.activities import create_inference_activity
            await create_inference_activity("model-42")
        finally:
            acts._create_uow = original

        call_args = mock_store.create.call_args[0][0]
        assert call_args.model_path == ""

    @pytest.mark.asyncio
    async def test_raises_when_engine_not_configured(self):
        """Activity raises RuntimeError if engine was never configured."""
        import interactors.temporal.activities as acts
        original_engine = acts._engine
        acts._engine = None
        try:
            from interactors.temporal.activities import create_inference_activity
            with pytest.raises(RuntimeError, match="Engine has not been configured"):
                await create_inference_activity("model-1")
        finally:
            acts._engine = original_engine

    @pytest.mark.asyncio
    async def test_returns_new_instance_id(self):
        mock_instance = MagicMock()
        mock_instance.id = "inst-xyz789"
        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        import interactors.temporal.activities as acts
        original = acts._create_uow
        acts._create_uow = lambda: _mock_uow(inference_store=mock_store)
        try:
            from interactors.temporal.activities import create_inference_activity
            result = await create_inference_activity("model-99")
        finally:
            acts._create_uow = original

        assert result == "inst-xyz789"

    @pytest.mark.asyncio
    async def test_propagates_store_exception(self):
        mock_store = MagicMock()
        mock_store.create.side_effect = RuntimeError("DB unavailable")

        import interactors.temporal.activities as acts
        original = acts._create_uow
        acts._create_uow = lambda: _mock_uow(inference_store=mock_store)
        try:
            from interactors.temporal.activities import create_inference_activity
            with pytest.raises(RuntimeError, match="DB unavailable"):
                await create_inference_activity("model-1")
        finally:
            acts._create_uow = original
