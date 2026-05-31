"""Unit tests for create_inference_activity."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestCreateInferenceActivity:
    """Tests for create_inference_activity."""

    @pytest.mark.asyncio
    async def test_creates_instance_with_correct_model_id(self):
        mock_instance = MagicMock()
        mock_instance.id = "inst-abc123"
        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        import interactors.temporal.activities as acts
        acts._inference_store = mock_store
        try:
            from interactors.temporal.activities import create_inference_activity
            await create_inference_activity("model-42")
        finally:
            acts._inference_store = None

        call_args = mock_store.create.call_args[0][0]
        assert call_args.model_id == "model-42"

    @pytest.mark.asyncio
    async def test_creates_instance_with_model_path(self):
        mock_instance = MagicMock()
        mock_instance.id = "inst-def456"
        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        import interactors.temporal.activities as acts
        acts._inference_store = mock_store
        try:
            from interactors.temporal.activities import create_inference_activity
            await create_inference_activity("model-42", "workflow/abc/model.gguf")
        finally:
            acts._inference_store = None

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
        acts._inference_store = mock_store
        try:
            from interactors.temporal.activities import create_inference_activity
            await create_inference_activity("model-42")
        finally:
            acts._inference_store = None

        call_args = mock_store.create.call_args[0][0]
        assert call_args.model_path == ""

    @pytest.mark.asyncio
    async def test_raises_when_store_not_configured(self):
        import interactors.temporal.activities as acts
        acts._inference_store = None
        from interactors.temporal.activities import create_inference_activity
        with pytest.raises(RuntimeError, match="InferenceStorePort has not been configured"):
            await create_inference_activity("model-1")

    @pytest.mark.asyncio
    async def test_returns_new_instance_id(self):
        mock_instance = MagicMock()
        mock_instance.id = "inst-xyz789"
        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        import interactors.temporal.activities as acts
        acts._inference_store = mock_store
        try:
            from interactors.temporal.activities import create_inference_activity
            result = await create_inference_activity("model-99")
        finally:
            acts._inference_store = None

        assert result == "inst-xyz789"

    @pytest.mark.asyncio
    async def test_propagates_store_exception(self):
        mock_store = MagicMock()
        mock_store.create.side_effect = RuntimeError("DB unavailable")

        import interactors.temporal.activities as acts
        acts._inference_store = mock_store
        try:
            from interactors.temporal.activities import create_inference_activity
            with pytest.raises(RuntimeError, match="DB unavailable"):
                await create_inference_activity("model-1")
        finally:
            acts._inference_store = None
