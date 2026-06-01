"""Unit tests for domain.train.export helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.train.export import _ensure_tokenizer_files


def _make_tokenizer_json(
    checkpoint: Path,
    merges: list[str] | list[list[str]] | None = None,
    vocab: dict | None = None,
) -> None:
    if merges is None:
        merges = ["Ġ t", "h e", "Ġt he", "i n", "in g"]
    if vocab is None:
        vocab = {"<|endoftext|>": 0, "Ġ": 1, "t": 2, "h": 3, "e": 4}
    data = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
        },
    }
    (checkpoint / "tokenizer.json").write_text(json.dumps(data))


class TestEnsureTokenizerFiles:
    def test_extracts_merges_and_vocab_from_tokenizer_json(self, tmp_path: Path) -> None:
        _make_tokenizer_json(tmp_path)

        _ensure_tokenizer_files(tmp_path)

        merges_file = tmp_path / "merges.txt"
        vocab_file = tmp_path / "vocab.json"
        assert merges_file.exists(), "merges.txt not created"
        assert vocab_file.exists(), "vocab.json not created"

    def test_merges_content_matches_tokenizer_json(self, tmp_path: Path) -> None:
        merges = ["Ġ t", "h e", "Ġt he"]
        _make_tokenizer_json(tmp_path, merges=merges)

        _ensure_tokenizer_files(tmp_path)

        written = (tmp_path / "merges.txt").read_text().splitlines()
        assert written == merges

    def test_vocab_content_matches_tokenizer_json(self, tmp_path: Path) -> None:
        vocab = {"<|endoftext|>": 0, "hello": 1, "world": 2}
        _make_tokenizer_json(tmp_path, vocab=vocab)

        _ensure_tokenizer_files(tmp_path)

        written = json.loads((tmp_path / "vocab.json").read_text())
        assert written == vocab

    def test_idempotent_when_files_already_exist(self, tmp_path: Path) -> None:
        _make_tokenizer_json(tmp_path)
        (tmp_path / "merges.txt").write_text("original content")
        (tmp_path / "vocab.json").write_text('{"existing": 0}')

        _ensure_tokenizer_files(tmp_path)

        assert (tmp_path / "merges.txt").read_text() == "original content"
        assert (tmp_path / "vocab.json").read_text() == '{"existing": 0}'

    def test_no_op_when_tokenizer_json_missing(self, tmp_path: Path) -> None:
        _ensure_tokenizer_files(tmp_path)

        assert not (tmp_path / "merges.txt").exists()
        assert not (tmp_path / "vocab.json").exists()

    def test_skips_non_bpe_tokenizer(self, tmp_path: Path) -> None:
        data = {"model": {"type": "WordPiece", "vocab": {"[PAD]": 0}}}
        (tmp_path / "tokenizer.json").write_text(json.dumps(data))

        _ensure_tokenizer_files(tmp_path)

        assert not (tmp_path / "merges.txt").exists()
        assert not (tmp_path / "vocab.json").exists()

    def test_extracts_only_vocab_when_merges_already_present(self, tmp_path: Path) -> None:
        _make_tokenizer_json(tmp_path)
        (tmp_path / "merges.txt").write_text("existing merges")

        _ensure_tokenizer_files(tmp_path)

        assert (tmp_path / "merges.txt").read_text() == "existing merges"
        assert (tmp_path / "vocab.json").exists()

    def test_extracts_only_merges_when_vocab_already_present(self, tmp_path: Path) -> None:
        _make_tokenizer_json(tmp_path)
        (tmp_path / "vocab.json").write_text('{"existing": 0}')

        _ensure_tokenizer_files(tmp_path)

        assert (tmp_path / "vocab.json").read_text() == '{"existing": 0}'
        assert (tmp_path / "merges.txt").exists()

    def test_merges_as_list_of_lists_is_handled(self, tmp_path: Path) -> None:
        # HuggingFace fast tokenizers (e.g. SmolLM) store merges as [["Ġ", "t"], ...]
        merges = [["Ġ", "t"], ["h", "e"], ["Ġt", "he"]]
        _make_tokenizer_json(tmp_path, merges=merges)

        _ensure_tokenizer_files(tmp_path)

        written = (tmp_path / "merges.txt").read_text().splitlines()
        assert written == ["Ġ t", "h e", "Ġt he"]

    def test_no_op_when_merges_empty_in_tokenizer_json(self, tmp_path: Path) -> None:
        _make_tokenizer_json(tmp_path, merges=[])

        _ensure_tokenizer_files(tmp_path)

        assert not (tmp_path / "merges.txt").exists()

    def test_unicode_vocab_round_trips_correctly(self, tmp_path: Path) -> None:
        vocab = {"Ġhello": 0, "世界": 1, "é": 2}
        _make_tokenizer_json(tmp_path, vocab=vocab)

        _ensure_tokenizer_files(tmp_path)

        written = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
        assert written == vocab
