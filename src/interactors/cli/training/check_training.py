"""CLI: check the status, progress, and logs of a running remote training job."""

from __future__ import annotations

import argparse
import sys


_BACKEND_PREFIXES = {
    "kaggle": "kaggle/",
    "ssh": "ssh/",
    "colab": "colab/",
}


def _detect_backend(run_id: str) -> str:
    for backend, prefix in _BACKEND_PREFIXES.items():
        if run_id.startswith(prefix):
            return backend
    if run_id.startswith("workflow/"):
        # k8s, runpod, and vastai all share the workflow/ namespace.
        # Read backend.txt written by submit() to identify which backend created this run.
        try:
            from adapters.storage.s3 import S3StorageAdapter
            return S3StorageAdapter().read_text(f"{run_id}/backend.txt").strip()
        except Exception:
            pass
    return ""


def _make_adapter(backend: str):
    if backend == "k8s":
        from adapters.compute.k8s.adapter import K8sTrainingAdapter
        return K8sTrainingAdapter()
    if backend == "runpod":
        from adapters.compute.runpod import RunPodTrainingAdapter
        return RunPodTrainingAdapter()
    if backend == "vastai":
        from adapters.compute.vastai.adapter import VastAiTrainingAdapter
        return VastAiTrainingAdapter()
    if backend == "kaggle":
        from adapters.compute.kaggle import KaggleTrainingAdapter
        return KaggleTrainingAdapter()
    if backend == "ssh":
        from adapters.compute.ssh import SshTrainingAdapter
        return SshTrainingAdapter()
    if backend == "colab":
        from adapters.compute.colab.adapter import ColabTrainingAdapter
        return ColabTrainingAdapter()
    raise SystemExit(f"ERROR: unknown backend {backend!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check the status, progress, and logs of a running remote training job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id", required=True, dest="run_id",
        help="Opaque run ID returned by trigger_training (e.g. k8s/my-exp-a1b2c3)",
    )
    parser.add_argument(
        "--backend", dest="backend", default="",
        choices=["", "k8s", "runpod", "vastai", "kaggle", "ssh", "colab"],
        help="Remote backend; auto-detected from the run_id prefix if omitted",
    )
    parser.add_argument(
        "--logs", action="store_true", default=False,
        help="Print recent instance logs (last ~200 lines)",
    )
    args = parser.parse_args(argv)

    backend = args.backend or _detect_backend(args.run_id)
    if not backend:
        print(
            f"ERROR: cannot detect backend from run_id {args.run_id!r}. "
            "Pass --backend explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    adapter = _make_adapter(backend)

    status = adapter.status(args.run_id)
    fraction, detail = adapter.progress(args.run_id)

    print(f"run_id   : {args.run_id}")
    print(f"backend  : {backend}")
    print(f"status   : {status}")
    print(f"progress : {fraction * 100:.0f}%  {detail}")

    if args.logs or status in ("running", "failed"):
        logs = adapter.logs(args.run_id)
        if logs:
            print("\n--- logs (last ~200 lines) ---")
            print(logs)
        else:
            print("\n(no logs available yet)")


if __name__ == "__main__":
    main()
