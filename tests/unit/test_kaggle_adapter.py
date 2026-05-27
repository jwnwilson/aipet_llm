"""Unit tests for KaggleTrainingAdapter.

Subprocess calls (kaggle CLI) and filesystem side-effects are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.models import RemoteTrainConfig


def _config(**kwargs) -> RemoteTrainConfig:
    defaults = dict(
        model="HuggingFaceTB/SmolLM2-360M",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
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
    def test_notebook_sets_storage_backend_to_kaggle(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "STORAGE_BACKEND" in source
        assert "'kaggle'" in source

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

    def test_notebook_sets_kaggle_data_dir(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "KAGGLE_DATA_DIR" in source
        assert "/kaggle/input/my-exp-data" in source

    def test_notebook_sets_train_and_eval_data_keys_as_filenames(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "TRAIN_DATA_KEY" in source
        assert "train.jsonl" in source
        assert "EVAL_DATA_KEY" in source
        assert "eval.jsonl" in source

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
