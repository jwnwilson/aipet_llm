"""Unit tests for KaggleTrainingAdapter.

Subprocess calls (kaggle CLI) and filesystem side-effects are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.models import RemoteTrainConfig


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    """Inject dummy Kaggle-specific AWS credentials so the _stage_dataset guard doesn't fire."""
    monkeypatch.setenv("KAGGLE_AWS_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("KAGGLE_AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")


def _config(**kwargs) -> RemoteTrainConfig:
    defaults = dict(
        model="HuggingFaceTB/SmolLM2-360M",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        train_s3_key="data/workflow/test-run/train.jsonl",
        eval_s3_key="data/workflow/test-run/eval.jsonl",
        run_id="test-run-id",
        epochs=1,
        patience=3,
        warmup_ratio=0.05,
        experiment_name="my-exp",
    )
    defaults.update(kwargs)
    return RemoteTrainConfig(**defaults)


def _make_adapter(tmp_path: Path, monkeypatch) -> "KaggleTrainingAdapter":
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    from adapters.compute.kaggle.adapter import KaggleTrainingAdapter
    return KaggleTrainingAdapter(work_dir=tmp_path)


class TestRenderNotebook:
    def test_notebook_sets_storage_backend_to_s3(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "STORAGE_BACKEND" in source
        assert "'s3'" in source
        assert "'kaggle'" not in source

    def test_notebook_invokes_remote_worker(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "remote_worker" in source
        # Must use subprocess so remote_worker starts in a fresh process with
        # packages already installed — matching K8s/RunPod behaviour.
        assert "subprocess" in source
        assert "run_module" not in source

    def test_notebook_passes_job_type_train(self, tmp_path, monkeypatch):
        """remote_worker dispatcher requires JOB_TYPE; training notebooks must pass 'train'."""
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "'JOB_TYPE': 'train'" in source

    def test_notebook_does_not_set_kaggle_data_dir(self, tmp_path, monkeypatch):
        """KAGGLE_DATA_DIR must not appear — training data comes from S3."""
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "KAGGLE_DATA_DIR" not in source

    def test_notebook_sets_train_and_eval_data_keys_as_s3_keys(self, tmp_path, monkeypatch):
        """TRAIN_DATA_KEY and EVAL_DATA_KEY must be the original S3 keys."""
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "TRAIN_DATA_KEY" in source
        assert "data/workflow/test-run/train.jsonl" in source
        assert "EVAL_DATA_KEY" in source
        assert "data/workflow/test-run/eval.jsonl" in source
        assert "S3_KEY_PREFIX" in source
        assert "workflow/test-run-id" in source

    def test_notebook_installs_wheel_and_training_deps(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        # Wheel is installed first; training deps are installed explicitly because
        # pip install /local/file.whl[extras] silently ignores extras on some pip versions.
        assert "pip" in source and "install" in source
        assert "transformers" in source
        assert "datasets" in source
        # No importlib cache hacks or domain-specific pre-imports — env setup only.
        assert "importlib" not in source
        assert "sys.modules" not in source


class TestDownload:
    def test_download_returns_checkpoint_dir_when_found(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        dest = tmp_path / "output"

        def fake_run(cmd, **kwargs):
            # Simulate kernel output writing checkpoint/
            (dest / "checkpoint").mkdir(parents=True, exist_ok=True)
            (dest / "checkpoint" / "config.json").write_text('{"model_type": "gpt2"}')

        with patch("subprocess.run", side_effect=fake_run):
            result = adapter.download("testuser/my-exp", dest)

        assert result == str(dest / "checkpoint")

    def test_download_falls_back_to_config_json_when_no_checkpoint_dir(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        dest = tmp_path / "output"

        def fake_run(cmd, **kwargs):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "config.json").write_text('{"model_type": "gpt2"}')

        with patch("subprocess.run", side_effect=fake_run):
            result = adapter.download("testuser/my-exp", dest)

        assert result == str(dest)

    def test_download_returns_eval_result_json_for_eval_jobs(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        dest = tmp_path / "output"

        def fake_run(cmd, **kwargs):
            # Eval kernel writes eval_result.json alongside a checkpoint dir
            (dest / "checkpoint").mkdir(parents=True, exist_ok=True)
            (dest / "eval_result.json").write_text('{"valid_pct": 0.95, "passed": true}')

        with patch("subprocess.run", side_effect=fake_run):
            result = adapter.download("testuser/my-exp-eval", dest)

        assert result == str(dest / "eval_result.json")


class TestSlugify:
    """Unit tests for the _slugify helper (Kaggle slug validation rules)."""

    def _slugify(self, name: str) -> str:
        from adapters.compute.kaggle.adapter import _slugify
        return _slugify(name)

    def test_normal_name_passes_through(self):
        assert self._slugify("my-experiment") == "my-experiment"

    def test_uppercase_is_lowercased(self):
        assert self._slugify("MyExp") == "myexp"

    def test_spaces_become_hyphens(self):
        assert self._slugify("my experiment") == "my-experiment"

    def test_special_chars_become_hyphens(self):
        assert self._slugify("foo@bar!baz") == "foo-bar-baz"

    def test_max_50_chars_enforced(self):
        long_name = "a" * 60
        result = self._slugify(long_name)
        assert len(result) <= 50

    def test_short_slug_padded_to_min_5_chars(self):
        # "ab" → 2 chars → should be padded to 5
        result = self._slugify("ab")
        assert len(result) >= 5
        assert result == "ab---"

    def test_single_char_slug_padded(self):
        result = self._slugify("x")
        assert len(result) >= 5
        assert result == "x----"

    def test_empty_string_falls_back_to_model_then_padded(self):
        # Empty after stripping → "model" (5 chars, exactly at minimum)
        result = self._slugify("---")
        assert result == "model"
        assert len(result) >= 5

    def test_exactly_5_chars_not_padded(self):
        result = self._slugify("hello")
        assert result == "hello"
        assert len(result) == 5
