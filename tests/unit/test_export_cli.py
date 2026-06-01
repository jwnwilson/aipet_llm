"""Unit tests for interactors.cli.training.export (the export CLI entrypoint)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


class TestExportCliDelegation:
    def test_calls_domain_export_with_correct_paths(self, tmp_path):
        ckpt = tmp_path / "checkpoint"
        out = tmp_path / "model.gguf"

        with patch("domain.train.export.export") as mock_export:
            from interactors.cli.training.export import main
            main(["--checkpoint", str(ckpt), "--output", str(out)])

        mock_export.assert_called_once_with(
            checkpoint=ckpt,
            output=out,
            quantize="Q4_K_M",
        )

    def test_passes_custom_quantize_type(self, tmp_path):
        with patch("domain.train.export.export") as mock_export:
            from interactors.cli.training.export import main
            main(["--checkpoint", "c", "--output", "o", "--quantize", "Q8_0"])
        _, kwargs = mock_export.call_args
        assert kwargs["quantize"] == "Q8_0"

    def test_passes_llama_cpp_dir_when_provided(self, tmp_path):
        llama_dir = tmp_path / "llama.cpp"
        with patch("domain.train.export.export") as mock_export:
            from interactors.cli.training.export import main
            main(["--checkpoint", "c", "--output", "o", "--llama-cpp-dir", str(llama_dir)])
        _, kwargs = mock_export.call_args
        assert kwargs["llama_cpp_dir"] == llama_dir

    def test_no_llama_cpp_dir_kwarg_when_omitted(self, tmp_path):
        with patch("domain.train.export.export") as mock_export:
            from interactors.cli.training.export import main
            main(["--checkpoint", "c", "--output", "o"])
        _, kwargs = mock_export.call_args
        assert "llama_cpp_dir" not in kwargs


class TestExportCliExitCodes:
    def test_propagates_system_exit_from_llama_cpp(self, tmp_path):
        """llama.cpp setup failures use sys.exit — the CLI must not swallow it."""
        with patch("domain.train.export.export", side_effect=SystemExit(1)):
            from interactors.cli.training.export import main
            with pytest.raises(SystemExit) as exc_info:
                main(["--checkpoint", "c", "--output", "o"])
        assert exc_info.value.code == 1

    def test_exit_2_on_unexpected_exception(self, tmp_path):
        with patch("domain.train.export.export", side_effect=RuntimeError("disk full")):
            from interactors.cli.training.export import main
            with pytest.raises(SystemExit) as exc_info:
                main(["--checkpoint", "c", "--output", "o"])
        assert exc_info.value.code == 2

    def test_exit_2_on_import_error(self, tmp_path):
        with patch.dict("sys.modules", {"domain.train.export": None}):
            from interactors.cli.training import export as export_mod
            import importlib
            importlib.reload(export_mod)
            with pytest.raises(SystemExit) as exc_info:
                export_mod.main(["--checkpoint", "c", "--output", "o"])
        assert exc_info.value.code == 2


class TestExportCliArgParsing:
    def test_missing_checkpoint_exits(self):
        from interactors.cli.training.export import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--output", "o.gguf"])
        assert exc_info.value.code != 0

    def test_missing_output_exits(self):
        from interactors.cli.training.export import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--checkpoint", "ckpt"])
        assert exc_info.value.code != 0
