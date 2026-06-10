"""Unit tests for the HF training progress callback (_ProgressCallback).

The callback owns two responsibilities:
  1. Write a `progress.json` sidecar that includes a *banded* `fraction` (so the
     RunPod/VastAI `progress()` adapters, which read `fraction`, advance the SSE
     stream) while keeping `step`/`max_steps` for the SSH/Kaggle adapters.
  2. Emit a clean periodic `log.info` line so the % complete reaches S3 `logs.txt`
     via the remote worker's logging buffer — on every log step AND on a time floor.
"""
import json
import logging
from types import SimpleNamespace

import pytest

from domain.train.trainer import _ProgressCallback


def _state(step: int, max_steps: int, epoch: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(global_step=step, max_steps=max_steps, epoch=epoch)


def test_emit_writes_banded_fraction(tmp_path):
    """fraction is mapped into [floor, floor+span]; step/max_steps kept for compat."""
    path = tmp_path / "progress.json"
    cb = _ProgressCallback(path, floor=0.15, span=0.75)

    cb._emit(_state(step=250, max_steps=500), force=True)

    data = json.loads(path.read_text())
    assert data["fraction"] == 0.525  # 0.15 + 0.75 * (250/500)
    assert data["step"] == 250
    assert data["max_steps"] == 500
    assert data["detail"]  # non-empty


def test_emit_zero_max_steps_does_not_divide_by_zero(tmp_path):
    path = tmp_path / "progress.json"
    cb = _ProgressCallback(path, floor=0.2, span=0.6)

    cb._emit(_state(step=0, max_steps=0), force=True)

    data = json.loads(path.read_text())
    assert data["fraction"] == 0.2  # equals floor, no error


def test_emit_logs_clean_percent_line(tmp_path):
    path = tmp_path / "progress.json"
    cb = _ProgressCallback(path)

    # Do NOT rely on pytest's caplog here. Under this repo's config
    # (--dist=loadfile + log_cli=true) on CI, caplog's handler level and global
    # logging state (logging.disable(), ancestor propagation) leak across tests in
    # a worker and silently drop the record — making this assertion fail CI-only.
    # Instead attach our own handler to the *exact* logger object _emit() writes to
    # and capture into our own list, fully independent of global logging state.
    from domain.train import trainer as trainer_mod

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    handler.setLevel(logging.INFO)

    logging.disable(logging.NOTSET)  # undo any leaked global suppression
    logger = trainer_mod.log  # the precise logger instance _emit() logs through
    prev_level, prev_propagate, prev_disabled = logger.level, logger.propagate, logger.disabled
    # logging.config.fileConfig(disable_existing_loggers=True) — used by Alembic's
    # env.py — sets `.disabled = True` on existing loggers; that flag is NOT cleared
    # by logging.disable()/setLevel(), so reset it explicitly or _emit() stays muted.
    logger.disabled = False
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        cb._emit(_state(step=42, max_steps=100), force=True)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate
        logger.disabled = prev_disabled

    lines = [r.getMessage() for r in records if "training progress" in r.getMessage()]
    assert len(lines) == 1
    assert "42%" in lines[0]


def test_on_step_end_throttles_to_time_floor(tmp_path):
    """on_step_end emits on first call, skips within min_interval_s, emits again after."""
    path = tmp_path / "progress.json"
    cb = _ProgressCallback(path, min_interval_s=30.0)

    clock = {"t": 1000.0}
    cb._now = lambda: clock["t"]
    emitted: list[float] = []
    real_emit = cb._emit
    cb._emit = lambda state, force=False: (emitted.append(clock["t"]), real_emit(state, force))[1]

    state = _state(step=10, max_steps=100)
    cb.on_step_end(None, state, None)        # t=1000 → first emit
    clock["t"] = 1010.0
    cb.on_step_end(None, state, None)        # +10s → throttled, no emit
    clock["t"] = 1040.0
    cb.on_step_end(None, state, None)        # +40s → emit

    assert emitted == [1000.0, 1040.0]


def test_on_log_caches_loss_for_later_time_floor_emit(tmp_path):
    """A loss seen in on_log is included in a subsequent time-floor emit."""
    path = tmp_path / "progress.json"
    cb = _ProgressCallback(path, min_interval_s=0.0)

    cb.on_log(None, _state(step=5, max_steps=100), None, logs={"loss": 1.2345})

    data = json.loads(path.read_text())
    assert data["loss"] == 1.2345
    assert "loss=1.2345" in data["detail"]
