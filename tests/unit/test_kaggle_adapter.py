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
        assert "run_module" in source

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

    def test_notebook_installs_wheel_with_training_extras(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        kernel_dir = tmp_path / "my-exp"
        kernel_dir.mkdir()
        adapter._render_notebook(_config(), kernel_dir, "my-exp-data")

        nb = json.loads((kernel_dir / "notebook.ipynb").read_text())
        source = "".join(nb["cells"][0]["source"])
        assert "[training]" in source


class TestEval:
    def test_eval_reads_result_from_kernel_output(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)

        # Simulate kernel output: eval_result.json in output dir
        output_dir = tmp_path / "my-exp-output"
        output_dir.mkdir()
        (output_dir / "eval_result.json").write_text('{"valid_pct": 0.97, "passed": true}')

        with patch("subprocess.run"):
            valid_pct, passed = adapter.eval("testuser/my-exp", "eval.jsonl")

        assert valid_pct == pytest.approx(0.97)
        assert passed is True

    def test_eval_reads_from_nested_working_dir(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)

        output_dir = tmp_path / "my-exp-output"
        nested = output_dir / "kaggle" / "working"
        nested.mkdir(parents=True)
        (nested / "eval_result.json").write_text('{"valid_pct": 0.8, "passed": false}')

        with patch("subprocess.run"):
            valid_pct, passed = adapter.eval("testuser/my-exp", "eval.jsonl")

        assert valid_pct == pytest.approx(0.8)
        assert passed is False

    def test_eval_raises_when_result_file_missing(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        # output_dir created but no eval_result.json inside
        (tmp_path / "my-exp-output").mkdir()

        with patch("subprocess.run"):
            with pytest.raises(RuntimeError, match="eval_result.json not found"):
                adapter.eval("testuser/my-exp", "eval.jsonl")


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
