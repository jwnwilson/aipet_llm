"""Kaggle-backed remote training adapter implementing RemoteTrainingPort."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from adapters.compute._wheel import build_wheel

from domain.models import RemoteTrainConfig
from domain.ports import RemoteTrainingPort

log = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Convert an arbitrary name to a Kaggle-safe slug (lowercase, hyphens, 6-50 chars)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:50] or "model"

def _kaggle_bin() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    candidate = Path(sys.executable).parent / "kaggle"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        "kaggle CLI not found in PATH or alongside Python interpreter. "
        "Install with: uv sync"
    )


_STATUS_MAP: dict[str, str] = {
    "complete": "done",
    "error": "failed",
    "failed": "failed",
    "running": "running",
    "queued": "pending",
    "cancelacknowledged": "failed",
}


class KaggleTrainingAdapter(RemoteTrainingPort):
    """RemoteTrainingPort implementation that submits training as a Kaggle kernel.

    Credentials are read from env vars ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.
    No internet access is required on the Kaggle kernel — the project is built
    into a wheel locally and uploaded as part of the dataset.

    Typical flow:
        1. Build a wheel of the project and stage it alongside the .jsonl data
           files as a Kaggle Dataset (all flat files, no subdirectories).
        2. Render ``notebook_template.ipynb`` with the run config.
        3. Write ``kernel-metadata.json`` pointing at the dataset.
        4. Push the kernel — Kaggle queues it for GPU execution.
        5. Poll ``kaggle kernels status`` until done/failed.
        6. Pull the checkpoint archive via ``kaggle kernels output``.
    """

    def __init__(self, work_dir: Path | None = None) -> None:
        self._username = os.environ.get("KAGGLE_USERNAME", "")
        self._work_dir = work_dir or Path("models/kaggle_kernels")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(__file__).parents[4].resolve()

    # ------------------------------------------------------------------
    # RemoteTrainingPort
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        dataset_slug = _slugify(f"{config.experiment_name}-data")
        dataset_ref = f"{self._username}/{dataset_slug}"

        staging = self._work_dir / dataset_slug
        self._stage_dataset(config, staging, dataset_slug)
        self._push_dataset(staging)
        self._wait_for_dataset(dataset_ref)

        kernel_slug = _slugify(config.experiment_name)
        kernel_dir = self._work_dir / kernel_slug
        kernel_dir.mkdir(parents=True, exist_ok=True)
        self._render_notebook(config, kernel_dir, dataset_slug)

        slug = f"{self._username}/{kernel_slug}"
        metadata = {
            "id": slug,
            "title": config.experiment_name,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [dataset_ref],
        }
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        subprocess.run(
            [_kaggle_bin(),"kernels", "push", "-p", str(kernel_dir), "--accelerator", config.gpu_type],
            check=True,
        )

        return slug

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        result = subprocess.run(
            [_kaggle_bin(),"kernels", "status", run_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        output = result.stdout.lower()
        for keyword, canonical in _STATUS_MAP.items():
            if keyword in output:
                return canonical  # type: ignore[return-value]
        return "pending"

    def logs(self, run_id: str) -> str:
        frac, detail = self.progress(run_id)
        if detail:
            return detail
        result = subprocess.run(
            [_kaggle_bin(),"kernels", "status", run_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()

    def progress(self, run_id: str) -> tuple[float, str]:
        """Return (completion_fraction, detail_string) from the training progress.json sidecar.

        The training notebook writes /kaggle/working/progress.json after each HF Trainer
        log step.  Falls back to (0.0, "") if the file is unavailable.
        """
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    [_kaggle_bin(),"kernels", "output", run_id, "-p", tmpdir, "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                matches = list(Path(tmpdir).glob("**/progress.json"))
                if matches:
                    data = json.loads(matches[0].read_text())
                    step = data.get("step", 0)
                    max_steps = data.get("max_steps", 1)
                    fraction = step / max_steps if max_steps else 0.0
                    epoch = data.get("epoch", "?")
                    elapsed = data.get("elapsed_s", "?")
                    parts = [f"step={step}/{max_steps}", f"epoch={epoch}", f"elapsed={elapsed}s"]
                    for key in ("loss", "eval_loss", "grad_norm"):
                        if key in data:
                            parts.append(f"{key}={data[key]:.4f}")
                    return fraction, "  ".join(parts)
        except Exception:
            pass
        return 0.0, ""

    def download(self, run_id: str, dest: Path) -> str:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [_kaggle_bin(), "kernels", "output", run_id, "-p", str(dest)],
            check=True,
        )
        # remote_worker writes to /kaggle/working/checkpoint/
        # kaggle kernels output preserves the path structure
        for checkpoint_dir in sorted(dest.rglob("checkpoint")):
            if checkpoint_dir.is_dir():
                return str(checkpoint_dir)
        # Fallback: find HF model config.json
        for config_path in sorted(dest.rglob("config.json")):
            if '"model_type"' in config_path.read_text():
                return str(config_path.parent)
        return str(dest)

    def eval(self, run_id: str, eval_data: str) -> tuple[float, bool]:  # noqa: ARG002
        """Read eval_result.json written by remote_worker inside the training kernel."""
        slug_name = _slugify(run_id.split("/")[-1])
        result_dir = self._work_dir / f"{slug_name}-output"
        result_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [_kaggle_bin(), "kernels", "output", run_id, "-p", str(result_dir)],
            check=True,
        )
        result_file = result_dir / "eval_result.json"
        if not result_file.exists():
            # Also check nested kaggle/working/ subdirectory
            nested = result_dir / "kaggle" / "working" / "eval_result.json"
            if nested.exists():
                result_file = nested
            else:
                raise RuntimeError(
                    f"eval_result.json not found in kernel output at {result_dir}"
                )
        result = json.loads(result_file.read_text())
        return float(result["valid_pct"]), bool(result["passed"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stage_dataset(
        self, config: RemoteTrainConfig, staging: Path, dataset_slug: str
    ) -> None:
        """Build a project wheel and stage it with the training data for Kaggle upload.

        All files are placed flat in the staging directory (no subdirectories) to
        avoid Kaggle CLI upload reliability issues with nested directory structures.
        """
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        # Build a wheel of the project and copy it into staging
        build_wheel(self._project_root, staging)

        # Copy flat .jsonl training data files
        train_data = Path(config.train_data)
        if not train_data.is_absolute():
            train_data = self._project_root / train_data
        for jsonl in train_data.parent.glob("*.jsonl"):
            shutil.copy2(jsonl, staging / jsonl.name)

        meta = {
            "title": f"{config.experiment_name} Training Data",
            "id": f"{self._username}/{dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    def _push_dataset(self, staging: Path) -> None:
        """Create the dataset on first run; add a new version on subsequent runs."""
        staged_files = [f.name for f in staging.iterdir() if f.is_file()]
        log.info("Staged files for upload: %s", staged_files)

        create_result = subprocess.run(
            [_kaggle_bin(),"datasets", "create", "-p", str(staging)],
            capture_output=True, text=True,
        )
        create_output = (create_result.stdout + create_result.stderr).strip()
        dataset_exists = create_result.returncode != 0 or "error" in create_output.lower()

        if not dataset_exists:
            log.info("Dataset created: %s", create_output)
        else:
            log.info("Dataset exists, uploading new version … (%s)", create_output)
            version_result = subprocess.run(
                [_kaggle_bin(),"datasets", "version", "-p", str(staging), "-m", "update"],
                capture_output=True, text=True,
            )
            version_output = (version_result.stdout + version_result.stderr).strip()
            if version_result.returncode != 0 or "error" in version_output.lower():
                raise RuntimeError(f"Dataset version upload failed:\n{version_output}")
            log.info("Dataset version uploaded: %s", version_output)

    def _wait_for_dataset(self, dataset_ref: str, timeout: int = 300, interval: int = 15) -> None:
        """Poll via the Kaggle Python API until a .whl is visible in the dataset."""
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        log.info("Polling for .whl in dataset %s …", dataset_ref)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = api.dataset_list_files(dataset_ref)
                if response and getattr(response, "files", None):
                    names = [f.name for f in response.files]
                    whl = [n for n in names if n.endswith(".whl")]
                    if whl:
                        log.info("Dataset ready — %s visible.", whl[0])
                        return
                    log.info("  visible files: %s — no .whl yet …", names[:6])
                else:
                    log.info("  no files visible yet …")
            except Exception as exc:
                log.warning("  poll error: %s", exc)
            time.sleep(interval)

        log.warning(".whl not confirmed in %s after %ds — proceeding anyway.", dataset_ref, timeout)

    def _render_notebook(
        self, config: RemoteTrainConfig, kernel_dir: Path, dataset_slug: str
    ) -> None:
        """Render a notebook that invokes remote_worker via runpy after installing the wheel."""
        data_dir = f"/kaggle/input/{dataset_slug}"

        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "import subprocess, sys, os, runpy, glob\n",
                        "\n",
                        "# Install project wheel with training extras\n",
                        f"whl = glob.glob('/kaggle/input/{dataset_slug}/*.whl')[0]\n",
                        "subprocess.run([sys.executable, '-m', 'pip', 'install', f'{whl}[training]'], check=True)\n",
                        "\n",
                        "# Set env vars consumed by remote_worker.py\n",
                        f"os.environ['RUN_ID'] = {config.experiment_name!r}\n",
                        "os.environ['TRAIN_DATA_KEY'] = 'train.jsonl'\n",
                        "os.environ['EVAL_DATA_KEY'] = 'eval.jsonl'\n",
                        f"os.environ['MODEL'] = {config.model!r}\n",
                        f"os.environ['EPOCHS'] = {str(config.epochs)!r}\n",
                        f"os.environ['PATIENCE'] = {str(config.patience)!r}\n",
                        f"os.environ['WARMUP_RATIO'] = {str(config.warmup_ratio)!r}\n",
                        "os.environ['STORAGE_BACKEND'] = 'kaggle'\n",
                        f"os.environ['KAGGLE_DATA_DIR'] = {data_dir!r}\n",
                        "\n",
                        "# Run the unified training worker (same code as K8s/RunPod/Vast.ai)\n",
                        "runpy.run_module('interactors.cli.training.remote_worker', run_name='__main__')\n",
                    ],
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None,
                }
            ],
        }
        (kernel_dir / "notebook.ipynb").write_text(json.dumps(notebook, indent=1))


