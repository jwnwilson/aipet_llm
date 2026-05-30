"""Unit tests for create_inference_activity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCreateInferenceActivity:
    """Tests for create_inference_activity."""

    @pytest.mark.asyncio
    async def test_creates_instance_with_correct_model_id(self):
        """Activity creates an InferenceInstance with the correct model_id."""
        mock_instance = MagicMock()
        mock_instance.id = "inst-abc123"

        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        with patch("interactors.api.deps.get_inference_store", return_value=mock_store):
            from interactors.temporal.activities import create_inference_activity
            result = await create_inference_activity("model-42")

        mock_store.create.assert_called_once()
        call_args = mock_store.create.call_args[0][0]
        assert call_args.model_id == "model-42"

    @pytest.mark.asyncio
    async def test_creates_instance_with_model_path(self):
        """Activity passes model_path through to InferenceInstanceConfig."""
        mock_instance = MagicMock()
        mock_instance.id = "inst-def456"

        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        with patch("interactors.api.deps.get_inference_store", return_value=mock_store):
            from interactors.temporal.activities import create_inference_activity
            result = await create_inference_activity("model-42", "workflow/abc/model.gguf")

        call_args = mock_store.create.call_args[0][0]
        assert call_args.model_id == "model-42"
        assert call_args.model_path == "workflow/abc/model.gguf"

    @pytest.mark.asyncio
    async def test_creates_instance_with_empty_model_path_by_default(self):
        """model_path defaults to empty string when not supplied."""
        mock_instance = MagicMock()
        mock_instance.id = "inst-ghi789"

        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        with patch("interactors.api.deps.get_inference_store", return_value=mock_store):
            from interactors.temporal.activities import create_inference_activity
            await create_inference_activity("model-42")

        call_args = mock_store.create.call_args[0][0]
        assert call_args.model_path == ""

    @pytest.mark.asyncio
    async def test_returns_new_instance_id(self):
        """Activity returns the id of the newly created InferenceInstance."""
        mock_instance = MagicMock()
        mock_instance.id = "inst-xyz789"

        mock_store = MagicMock()
        mock_store.create.return_value = mock_instance

        with patch("interactors.api.deps.get_inference_store", return_value=mock_store):
            from interactors.temporal.activities import create_inference_activity
            result = await create_inference_activity("model-99")

        assert result == "inst-xyz789"

    @pytest.mark.asyncio
    async def test_propagates_store_exception(self):
        """If the store raises, the activity propagates the exception."""
        mock_store = MagicMock()
        mock_store.create.side_effect = RuntimeError("DB unavailable")

        with patch("interactors.api.deps.get_inference_store", return_value=mock_store):
            from interactors.temporal.activities import create_inference_activity
            with pytest.raises(RuntimeError, match="DB unavailable"):
                await create_inference_activity("model-1")
