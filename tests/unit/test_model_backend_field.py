"""Test backend and backend_model_id fields on TrainingModelConfig."""

from domain.models import TrainingModelConfig


def test_backend_defaults_to_local():
    """Test that backend defaults to 'local'."""
    cfg = TrainingModelConfig(name="test")
    assert cfg.backend == "local"
    assert cfg.backend_model_id == ""


def test_backend_can_be_set_to_openrouter():
    """Test that backend can be set to 'openrouter'."""
    cfg = TrainingModelConfig(name="test", backend="openrouter", backend_model_id="gpt-4")
    assert cfg.backend == "openrouter"
    assert cfg.backend_model_id == "gpt-4"
