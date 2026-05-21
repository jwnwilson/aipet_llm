"""Idle inference instance shutdown background task."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from domain.models import InferenceStatus
from domain.ports import InferenceStorePort, PodLifecyclePort

log = logging.getLogger(__name__)

_DEFAULT_IDLE_TIMEOUT_HOURS = 2
_POLL_INTERVAL_SECONDS = 300  # 5 minutes


def _idle_timeout_hours() -> int:
    try:
        return int(os.environ.get("INFERENCE_IDLE_TIMEOUT_HOURS", _DEFAULT_IDLE_TIMEOUT_HOURS))
    except ValueError:
        return _DEFAULT_IDLE_TIMEOUT_HOURS


def sweep_idle_instances(store: InferenceStorePort, pod_adapter: PodLifecyclePort) -> int:
    """Stop every active inference instance whose last activity exceeds the idle timeout.

    Returns the number of instances that were stopped.
    Per-instance errors are logged but do not propagate — the sweep continues.
    """
    timeout_hours = _idle_timeout_hours()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=timeout_hours)
    stopped = 0

    for instance in store.list_active():
        # Use last_used_at if set, otherwise fall back to created_at
        last_activity = instance.last_used_at or instance.created_at
        if last_activity > cutoff:
            continue

        log.info(
            "Stopping idle instance %s (last_activity=%s, cutoff=%s)",
            instance.id,
            last_activity,
            cutoff,
        )
        try:
            pod_adapter.delete_pod(
                pod_name=instance.pod_name,
                namespace=instance.pod_namespace,
            )
        except Exception:
            log.exception("Failed to delete pod for idle instance %s — marking failed", instance.id)
            store.update_status(instance.id, InferenceStatus.FAILED)
            continue

        store.update_status(instance.id, InferenceStatus.SHUTDOWN)
        stopped += 1

    return stopped


async def idle_shutdown_loop(store: InferenceStorePort, pod_adapter: PodLifecyclePort) -> None:
    """Long-running asyncio task that sweeps idle instances every POLL_INTERVAL_SECONDS."""
    poll_interval = int(os.environ.get("INFERENCE_IDLE_POLL_SECONDS", _POLL_INTERVAL_SECONDS))
    while True:
        await asyncio.sleep(poll_interval)
        try:
            stopped = sweep_idle_instances(store, pod_adapter)
            if stopped:
                log.info("Idle shutdown sweep stopped %d instance(s)", stopped)
        except Exception:
            log.exception("Idle shutdown sweep encountered an unexpected error")
