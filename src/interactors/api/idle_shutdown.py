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
_INITIALIZING_TIMEOUT_MINUTES = 5


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


_READINESS_POLL_SECONDS = 10


def check_initializing_instances(
    store: InferenceStorePort, pod_adapter: PodLifecyclePort
) -> int:
    """Promote INITIALIZING inference instances to AVAILABLE once their pod is Running.

    Returns the count of instances promoted to AVAILABLE.
    Per-instance errors are logged but do not propagate — the sweep continues.
    """
    promoted = 0
    for instance in store.list_active():
        if instance.status != InferenceStatus.INITIALIZING:
            continue
        try:
            phase = pod_adapter.pod_status(
                pod_name=instance.pod_name,
                namespace=instance.pod_namespace,
            )
        except Exception:
            log.exception("Failed to poll pod status for instance %s", instance.id)
            continue

        if phase == "running":
            store.update_status(instance.id, InferenceStatus.AVAILABLE)
            log.info("Instance %s pod running — marked AVAILABLE", instance.id)
            promoted += 1
        elif phase == "failed":
            store.update_status(instance.id, InferenceStatus.FAILED)
            log.warning("Instance %s pod failed — marked FAILED", instance.id)
        elif phase == "unknown":
            age = datetime.now(timezone.utc) - instance.updated_at
            if age > timedelta(minutes=_INITIALIZING_TIMEOUT_MINUTES):
                store.update_status(instance.id, InferenceStatus.FAILED)
                log.warning(
                    "Instance %s stuck INITIALIZING for %s with no pod — marked FAILED",
                    instance.id, age,
                )

    return promoted


async def readiness_watch_loop(
    store: InferenceStorePort, pod_adapter: PodLifecyclePort
) -> None:
    """Periodically promote INITIALIZING instances to AVAILABLE.

    Runs every INFERENCE_READINESS_POLL_SECONDS (default 10 s) so inference
    pods become usable within one poll cycle of their readiness probe passing.
    """
    poll_interval = int(
        os.environ.get("INFERENCE_READINESS_POLL_SECONDS", _READINESS_POLL_SECONDS)
    )
    while True:
        await asyncio.sleep(poll_interval)
        try:
            promoted = check_initializing_instances(store, pod_adapter)
            if promoted:
                log.info("Readiness sweep promoted %d instance(s) to AVAILABLE", promoted)
        except Exception:
            log.exception("Readiness watch loop encountered an unexpected error")
