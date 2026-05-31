"""Integration test: export() produces a loadable GGUF from an HF checkpoint.

Skips automatically unless:
  - llama.cpp is built  (llama.cpp/build/bin/llama-quantize present)
  - An HF checkpoint exists  (LLM_API_TEST_CHECKPOINT_PATH env var, or data/test_checkpoint/)
  - llama-cpp-python is installed  (inference extra)

Run locally once llama.cpp is built and a checkpoint is available:
    LLM_API_TEST_CHECKPOINT_PATH=data/workflow/<run_id>/checkpoint \
        uv run pytest tests/integration/test_export_pipeline.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_LLAMA_QUANTIZE = _REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize"
_LLAMA_CPP_DIR = _REPO_ROOT / "llama.cpp"

_DEFAULT_CHECKPOINT = _REPO_ROOT / "data" / "test_checkpoint"
_CHECKPOINT_PATH = Path(
    os.environ.get("LLM_API_TEST_CHECKPOINT_PATH", str(_DEFAULT_CHECKPOINT))
)

pytestmark = pytest.mark.skipif(
    not _LLAMA_QUANTIZE.exists(),
    reason=f"llama.cpp not built at {_LLAMA_QUANTIZE}",
)


@pytest.fixture(scope="module")
def checkpoint_path() -> Path:
    if not _CHECKPOINT_PATH.exists():
        pytest.skip(
            f"No HF checkpoint found at {_CHECKPOINT_PATH}. "
            "Set LLM_API_TEST_CHECKPOINT_PATH to a merged checkpoint directory."
        )
    return _CHECKPOINT_PATH


def test_export_produces_gguf(checkpoint_path: Path, tmp_path: Path) -> None:
    """export() creates the quantised GGUF file."""
    from domain.train.export import export

    output = tmp_path / "model.gguf"
    export(checkpoint=checkpoint_path, output=output, llama_cpp_dir=_LLAMA_CPP_DIR)

    assert output.exists(), "GGUF file not created"
    assert output.stat().st_size > 0, "GGUF file is empty"


def test_export_gguf_loads_in_adapter(checkpoint_path: Path, tmp_path: Path) -> None:
    """The exported GGUF loads without error in LlamaCppInferenceAdapter."""
    pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")

    from domain.train.export import export
    from adapters.inference import LlamaCppInferenceAdapter

    output = tmp_path / "model.gguf"
    export(checkpoint=checkpoint_path, output=output, llama_cpp_dir=_LLAMA_CPP_DIR)

    adapter = LlamaCppInferenceAdapter(model_path=str(output))
    # load() triggers llama_cpp.Llama() — this is where the tokenizer-merges
    # bug would have caused ValueError before the fix.
    adapter.load()
    adapter.release()


def test_export_extracts_tokenizer_merges_before_conversion(
    checkpoint_path: Path, tmp_path: Path
) -> None:
    """Merges extracted from tokenizer.json survive into the checkpoint dir during export.

    Regression test for the SmolLM BPE tokenizer bug: convert_hf_to_gguf.py
    reads merges from merges.txt (slow-tokenizer format). When only tokenizer.json
    is present, older converters silently omit merges and produce an unloadable GGUF.
    """
    import shutil
    from domain.train.export import export

    # Work on a copy so we don't modify the shared fixture
    checkpoint_copy = tmp_path / "checkpoint"
    shutil.copytree(checkpoint_path, checkpoint_copy)

    # Remove any pre-existing merges.txt to simulate a fast-tokenizer-only checkpoint
    merges_file = checkpoint_copy / "merges.txt"
    merges_file.unlink(missing_ok=True)

    output = tmp_path / "model.gguf"
    export(checkpoint=checkpoint_copy, output=output, llama_cpp_dir=_LLAMA_CPP_DIR)

    # After export, merges.txt must exist (extracted from tokenizer.json)
    # and the GGUF must have been created successfully
    assert merges_file.exists(), "merges.txt was not materialised from tokenizer.json"
    assert output.exists(), "GGUF file not created"
