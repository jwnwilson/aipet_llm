"""Unit tests for adapters.storage.kaggle_local.KaggleLocalStorageAdapter."""
from __future__ import annotations

from pathlib import Path

import pytest

from adapters.storage.kaggle_local import KaggleLocalStorageAdapter


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "input" / "my-exp-data"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "working"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def adapter(data_dir, work_dir):
    return KaggleLocalStorageAdapter(data_dir=data_dir, work_dir=work_dir)


class TestKaggleLocalStorageDownload:
    def test_download_copies_file_from_data_dir(self, adapter, data_dir, tmp_path):
        (data_dir / "train.jsonl").write_text('{"prompt":"hi"}')
        dest = tmp_path / "out" / "train.jsonl"
        adapter.download("train.jsonl", dest)
        assert dest.read_text() == '{"prompt":"hi"}'

    def test_download_raises_when_file_missing(self, adapter, tmp_path):
        with pytest.raises(FileNotFoundError):
            adapter.download("missing.jsonl", tmp_path / "out.jsonl")

    def test_download_directory_copies_all_files(self, adapter, data_dir, tmp_path):
        (data_dir / "train.jsonl").write_text("train")
        (data_dir / "eval.jsonl").write_text("eval")
        dest = tmp_path / "dest"
        adapter.download_directory("", dest)
        assert (dest / "train.jsonl").read_text() == "train"
        assert (dest / "eval.jsonl").read_text() == "eval"


class TestKaggleLocalStorageWrite:
    def test_write_bytes_creates_file_under_work_dir(self, adapter, work_dir):
        adapter.write_bytes("status.txt", b"running")
        assert (work_dir / "status.txt").read_bytes() == b"running"

    def test_write_bytes_creates_subdirectory(self, adapter, work_dir):
        adapter.write_bytes("subdir/progress.json", b'{"fraction": 0.5}')
        assert (work_dir / "subdir" / "progress.json").exists()

    def test_upload_copies_local_file_to_work_dir(self, adapter, work_dir, tmp_path):
        src = tmp_path / "model.bin"
        src.write_bytes(b"\x00\x01\x02")
        adapter.upload(src, "checkpoint/model.bin")
        assert (work_dir / "checkpoint" / "model.bin").read_bytes() == b"\x00\x01\x02"

    def test_read_text_reads_from_work_dir(self, adapter, work_dir):
        (work_dir / "eval_result.json").write_text('{"valid_pct": 0.97}')
        assert '"valid_pct"' in adapter.read_text("eval_result.json")

    def test_read_text_returns_empty_string_when_missing(self, adapter):
        assert adapter.read_text("nonexistent.txt") == ""


class TestKaggleLocalStorageExists:
    def test_exists_true_for_present_file_in_work_dir(self, adapter, work_dir):
        (work_dir / "status.txt").write_text("done")
        assert adapter.exists("status.txt") is True

    def test_exists_false_for_missing_file(self, adapter):
        assert adapter.exists("not_there.txt") is False
