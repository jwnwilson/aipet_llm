"""Kaggle-backed remote job adapter implementing RemoteJobPort."""

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

from domain.models import EvalJobSpec, RemoteJobSpec, TrainJobSpec
from domain.ports import RemoteJobPort

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


class KaggleTrainingAdapter(RemoteJobPort):
    """RemoteJobPort implementation that submits jobs as Kaggle kernels.

    Credentials are read from env vars ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.
    No internet access is required on the Kaggle kernel — the project is built
    into a wheel locally and uploaded as part of the dataset.

    Training flow (job_type="train"):
        1. Build a wheel and stage it with .jsonl data as a Kaggle Dataset.
        2. Render ``notebook_template.ipynb`` with the run config.
        3. Write ``kernel-metadata.json`` pointing at the dataset.
        4. Push the kernel — Kaggle queues it for GPU execution.
        5. Poll ``kaggle kernels status`` until done/failed (Temporal activity).
        6. Pull the checkpoint archive via ``kaggle kernels output``.

    Eval flow (job_type="eval"):
        1. Render ``eval_notebook_template.ipynb`` with training artifact ref.
        2. Push the eval kernel (references same dataset as training kernel).
        3. Poll via Temporal activity (no blocking loop in adapter).
        4. Pull ``eval_result.json`` from kernel output.
    """

    def __init__(self, work_dir: Path | None = None) -> None:
        self._username = os.environ.get("KAGGLE_USERNAME", "")
        self._work_dir = work_dir or Path("models/kaggle_kernels")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(__file__).parents[4].resolve()

    # ------------------------------------------------------------------
    # RemoteJobPort
    # ------------------------------------------------------------------

    def submit(self, spec: RemoteJobSpec) -> str:
        if isinstance(spec, TrainJobSpec):
            return self._submit_train(spec)
        if isinstance(spec, EvalJobSpec):
            return self._submit_eval(spec)
        raise NotImplementedError(f"Unsupported job_type: {spec.job_type!r}")

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        result = subprocess.run(
            [_kaggle_bin(), "kernels", "status", run_id],
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

    def logs(self, run_id: str) -> str:
        frac, detail = self.progress(run_id)
        if detail:
            return detail
        result = subprocess.run(
            [_kaggle_bin(), "kernels", "status", run_id],
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
                    [_kaggle_bin(), "kernels", "output", run_id, "-p", tmpdir, "--quiet"],
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _submit_train(self, config: TrainJobSpec) -> str:
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
            [_kaggle_bin(), "kernels", "push", "-p", str(kernel_dir), "--accelerator", config.gpu_type],
            check=True,
        )
        return slug

    def _submit_eval(self, spec: EvalJobSpec) -> str:
        """Push an eval kernel that downloads checkpoint from the training dataset and scores it.

        The blocking poll loop from the old ``eval()`` method is intentionally removed.
        Temporal activities poll ``status()`` in the same async loop used for training.
        ``training_artifact_ref`` is the training kernel slug (e.g. "username/exp-slug").
        """
        training_kernel_slug = spec.training_artifact_ref
        # Derive dataset ref from the training run slug (same dataset used for training).
        experiment_name = _slugify(training_kernel_slug.split("/")[-1])
        dataset_ref = f"{self._username}/{experiment_name}-data"

        eval_kernel_id = f"{experiment_name}-eval"
        eval_slug = f"{self._username}/{eval_kernel_id}"
        kernel_dir = self._work_dir / eval_kernel_id
        kernel_dir.mkdir(parents=True, exist_ok=True)

        self._render_eval_notebook(
            training_run_id=training_kernel_slug,
            eval_data=spec.eval_data,
            experiment_name=experiment_name,
            kernel_dir=kernel_dir,
        )

        metadata = {
            "id": eval_slug,
            "title": eval_kernel_id,
            "code_file": "eval_notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [dataset_ref],
        }
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        subprocess.run(
            [_kaggle_bin(), "kernels", "push", "-p", str(kernel_dir), "--accelerator", spec.gpu_type],
            check=True,
        )
        return eval_slug   # Temporal polls status(eval_slug) — no blocking loop here

    def _stage_dataset(
        self, config: TrainJobSpec, staging: Path, dataset_slug: str
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

        # Copy flat .jsonl training data files.
        # Resolve relative paths from CWD (where the Temporal worker runs), NOT from
        # _project_root — that resolves to the Python site-packages dir when the
        # project is installed as a wheel, which is never where train.jsonl lives.
        train_data = Path(config.train_data)
        if not train_data.is_absolute():
            train_data = Path.cwd() / train_data

        # Fail early with a clear message rather than uploading an empty dataset
        # and discovering the problem deep inside the Kaggle kernel.
        missing = [p for p in (train_data, train_data.parent / "eval.jsonl") if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Training data not found before Kaggle upload: {[str(p) for p in missing]}. "
                f"Ensure generate_dataset_activity ran successfully before train_activity "
                f"(cwd={Path.cwd()}, configured train_data={config.train_data!r})."
            )

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
            [_kaggle_bin(), "datasets", "create", "-p", str(staging)],
            capture_output=True, text=True,
        )
        create_output = (create_result.stdout + create_result.stderr).strip()
        dataset_exists = create_result.returncode != 0 or "error" in create_output.lower()

        if not dataset_exists:
            log.info("Dataset created: %s", create_output)
        else:
            log.info("Dataset exists, uploading new version … (%s)", create_output)
            version_result = subprocess.run(
                [_kaggle_bin(), "datasets", "version", "-p", str(staging), "-m", "update"],
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
        self, config: TrainJobSpec, kernel_dir: Path, dataset_slug: str
    ) -> None:
        """Render a notebook that invokes remote_worker via runpy after installing the wheel."""
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
                        "import subprocess, sys, os, glob, pathlib\n",
                        "\n",
                        "# Locate the project wheel — handle both old (/kaggle/input/<slug>/) and\n",
                        "# new (/kaggle/input/datasets/<owner>/<slug>/) Kaggle mount paths.\n",
                        f"_whl_list = (\n",
                        f"    glob.glob('/kaggle/input/{dataset_slug}/*.whl') or\n",
                        f"    glob.glob('/kaggle/input/**/{dataset_slug}/*.whl', recursive=True)\n",
                        ")\n",
                        "if not _whl_list:\n",
                        f'    raise FileNotFoundError("No .whl found for {dataset_slug!r} — re-trigger training to rebuild the dataset")\n',
                        "whl = _whl_list[0]\n",
                        "# Derive data dir from the wheel — correct regardless of Kaggle mount convention.\n",
                        "_data_dir = str(pathlib.Path(whl).parent)\n",
                        "\n",
                        "# Install the project wheel (provides domain/adapter code).\n",
                        "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', whl], check=True)\n",
                        "# pip install /local/file.whl[extras] silently ignores extras on some pip versions.\n",
                        "# Explicitly install training deps not pre-installed by Kaggle (torch is already present).\n",
                        "subprocess.run([\n",
                        "    sys.executable, '-m', 'pip', 'install', '--quiet',\n",
                        "    'transformers', 'datasets', 'accelerate', 'sentencepiece', 'peft', 'bitsandbytes',\n",
                        "], check=True)\n",
                        "\n",
                        "# Run the training worker in a fresh subprocess so it starts with all packages\n",
                        "# already installed — identical to K8s/RunPod where packages are pre-installed\n",
                        "# in the image. The subprocess inherits os.environ set below.\n",
                        "subprocess.run(\n",
                        "    [sys.executable, '-m', 'interactors.cli.training.remote_worker'],\n",
                        "    check=True,\n",
                        "    env={\n",
                        "        **os.environ,\n",
                        f"        'RUN_ID': {config.experiment_name!r},\n",
                        "        'TRAIN_DATA_KEY': 'train.jsonl',\n",
                        "        'EVAL_DATA_KEY': 'eval.jsonl',\n",
                        f"        'MODEL': {config.model!r},\n",
                        f"        'EPOCHS': {str(config.epochs)!r},\n",
                        f"        'PATIENCE': {str(config.patience)!r},\n",
                        f"        'WARMUP_RATIO': {str(config.warmup_ratio)!r},\n",
                        "        'STORAGE_BACKEND': 'kaggle',\n",
                        "        'KAGGLE_DATA_DIR': _data_dir,\n",
                        "    },\n",
                        ")\n",
                    ],
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None,
                }
            ],
        }
        (kernel_dir / "notebook.ipynb").write_text(json.dumps(notebook, indent=1))

    def _render_eval_notebook(
        self, training_run_id: str, eval_data: str, experiment_name: str, kernel_dir: Path
    ) -> None:
        template_path = Path(__file__).parent / "eval_notebook_template.ipynb"
        notebook = json.loads(template_path.read_text())

        config_repr = repr({
            "training_run_id": training_run_id,
            "experiment_name": experiment_name,
            "eval_data_file": Path(eval_data).name,
            "dataset_slug": _slugify(f"{experiment_name}-data"),
        })

        replacements = {"{{config}}": config_repr}
        for cell in notebook["cells"]:
            src = cell["source"]
            if isinstance(src, str):
                cell["source"] = _replace_all(src, replacements)
            else:
                cell["source"] = [_replace_all(line, replacements) for line in src]

        (kernel_dir / "eval_notebook.ipynb").write_text(json.dumps(notebook, indent=1))


def _replace_all(s: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s
