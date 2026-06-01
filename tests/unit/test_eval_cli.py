"""Unit tests for interactors.cli.training.eval (the eval CLI entrypoint)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(tmp_path: Path, argv: list[str], monkeypatch=None):
    """Import and call main(); capture SystemExit."""
    from interactors.cli.training.eval import main
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    return exc_info.value.code


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvalCliResultJson:
    def test_writes_results_json_on_success(self, tmp_path):
        out = tmp_path / "eval_results.json"
        eval_data = tmp_path / "eval.jsonl"
        eval_data.write_text('{"prompt":"p","response":"r"}\n')

        with (
            patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
            patch("domain.train.evaluate.infer_hf", return_value="IDLE"),
            patch("domain.train.evaluate.evaluate", return_value=(0, 0.97)) as mock_eval,
        ):
            from interactors.cli.training.eval import main
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "--checkpoint", str(tmp_path / "ckpt"),
                    "--eval-data", str(eval_data),
                    "--output", str(out),
                ])

        assert exc_info.value.code == 0
        result = json.loads(out.read_text())
        assert result["valid_pct"] == pytest.approx(0.97)
        assert result["passed"] is True

    def test_writes_results_json_when_below_threshold(self, tmp_path):
        out = tmp_path / "eval_results.json"
        eval_data = tmp_path / "eval.jsonl"
        eval_data.write_text('{"prompt":"p","response":"r"}\n')

        with (
            patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
            patch("domain.train.evaluate.infer_hf", return_value="IDLE"),
            patch("domain.train.evaluate.evaluate", return_value=(1, 0.80)),
        ):
            from interactors.cli.training.eval import main
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "--checkpoint", str(tmp_path / "ckpt"),
                    "--eval-data", str(eval_data),
                    "--output", str(out),
                ])

        assert exc_info.value.code == 1
        result = json.loads(out.read_text())
        assert result["valid_pct"] == pytest.approx(0.80)
        assert result["passed"] is False


class TestEvalCliExitCodes:
    def test_exit_0_when_domain_returns_exit_code_0(self, tmp_path):
        out = tmp_path / "out.json"
        with (
            patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
            patch("domain.train.evaluate.infer_hf", return_value="IDLE"),
            patch("domain.train.evaluate.evaluate", return_value=(0, 0.97)),
        ):
            from interactors.cli.training.eval import main
            with pytest.raises(SystemExit) as exc_info:
                main(["--checkpoint", "ckpt", "--eval-data", "e.jsonl", "--output", str(out)])
        assert exc_info.value.code == 0

    def test_exit_1_when_domain_returns_exit_code_1(self, tmp_path):
        out = tmp_path / "out.json"
        with (
            patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
            patch("domain.train.evaluate.infer_hf", return_value="IDLE"),
            patch("domain.train.evaluate.evaluate", return_value=(1, 0.80)),
        ):
            from interactors.cli.training.eval import main
            with pytest.raises(SystemExit) as exc_info:
                main(["--checkpoint", "ckpt", "--eval-data", "e.jsonl", "--output", str(out)])
        assert exc_info.value.code == 1

    def test_exit_2_on_import_error(self, tmp_path):
        out = tmp_path / "out.json"
        with patch.dict("sys.modules", {"domain.train.evaluate": None}):
            from interactors.cli.training import eval as eval_mod
            import importlib
            importlib.reload(eval_mod)
            with pytest.raises(SystemExit) as exc_info:
                eval_mod.main(["--checkpoint", "c", "--eval-data", "e", "--output", str(out)])
        assert exc_info.value.code == 2

    def test_exit_2_on_runtime_error(self, tmp_path):
        out = tmp_path / "out.json"
        with (
            patch("domain.train.evaluate.load_hf_pipeline", side_effect=RuntimeError("GPU OOM")),
        ):
            from interactors.cli.training.eval import main
            with pytest.raises(SystemExit) as exc_info:
                main(["--checkpoint", "c", "--eval-data", "e", "--output", str(out)])
        assert exc_info.value.code == 2


class TestEvalCliArgParsing:
    def test_missing_checkpoint_exits(self, tmp_path):
        from interactors.cli.training.eval import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--eval-data", "e.jsonl"])
        assert exc_info.value.code != 0

    def test_missing_eval_data_exits(self, tmp_path):
        from interactors.cli.training.eval import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--checkpoint", "ckpt"])
        assert exc_info.value.code != 0

    def test_default_output_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with (
            patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
            patch("domain.train.evaluate.infer_hf", return_value="IDLE"),
            patch("domain.train.evaluate.evaluate", return_value=(0, 0.95)),
        ):
            from interactors.cli.training.eval import main
            with pytest.raises(SystemExit):
                main(["--checkpoint", "ckpt", "--eval-data", "e.jsonl"])
        assert (tmp_path / "eval_results.json").exists()
