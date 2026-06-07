"""Unit tests for the idle inference shutdown sweep."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from domain.models import InferenceInstance, InferenceStatus
from interactors.api.idle_shutdown import sweep_idle_instances


def _make_instance(
    id: str = "inst-1",
    pod_name: str = "pod-1",
    pod_namespace: str = "default",
    status: InferenceStatus = InferenceStatus.AVAILABLE,
    last_used_at: datetime | None = None,
    created_at: datetime | None = None,
) -> InferenceInstance:
    now = datetime.now(timezone.utc)
    return InferenceInstance(
        id=id,
        model_id="model-1",
        pod_name=pod_name,
        pod_namespace=pod_namespace,
        status=status,
        idle_timeout_minutes=120,
        last_used_at=last_used_at,
        created_at=created_at or now,
        updated_at=now,
    )


def _make_store(instances: list[InferenceInstance]):
    store = MagicMock()
    store.list_active.return_value = instances
    store.update_status.return_value = MagicMock()
    return store


def _make_pod():
    pod = MagicMock()
    pod.delete_pod.return_value = None
    return pod


class TestSweepIdleInstances:
    def test_returns_zero_when_no_active_instances(self):
        store = _make_store([])
        pod = _make_pod()
        stopped = sweep_idle_instances(store, pod)
        assert stopped == 0
        pod.delete_pod.assert_not_called()

    def test_skips_recently_used_instance(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_IDLE_TIMEOUT_HOURS", "2")
        recent = _make_instance(
            last_used_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        )
        store = _make_store([recent])
        pod = _make_pod()

        stopped = sweep_idle_instances(store, pod)

        assert stopped == 0
        pod.delete_pod.assert_not_called()
        store.update_status.assert_not_called()

    def test_stops_instance_idle_past_timeout(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_IDLE_TIMEOUT_HOURS", "2")
        idle = _make_instance(
            id="inst-idle",
            pod_name="pod-idle",
            pod_namespace="prod",
            last_used_at=datetime.now(timezone.utc) - timedelta(hours=3),
        )
        store = _make_store([idle])
        pod = _make_pod()

        stopped = sweep_idle_instances(store, pod)

        assert stopped == 1
        pod.delete_pod.assert_called_once_with(pod_name="pod-idle", namespace="prod")
        store.update_status.assert_called_once_with("inst-idle", InferenceStatus.SHUTDOWN)

    def test_uses_created_at_when_last_used_at_is_none(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_IDLE_TIMEOUT_HOURS", "2")
        old_instance = _make_instance(
            id="inst-old",
            last_used_at=None,
            created_at=datetime.now(timezone.utc) - timedelta(hours=5),
        )
        store = _make_store([old_instance])
        pod = _make_pod()

        stopped = sweep_idle_instances(store, pod)

        assert stopped == 1
        store.update_status.assert_called_once_with("inst-old", InferenceStatus.SHUTDOWN)

    def test_skips_new_instance_with_no_last_used(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_IDLE_TIMEOUT_HOURS", "2")
        new_instance = _make_instance(
            last_used_at=None,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        store = _make_store([new_instance])
        pod = _make_pod()

        stopped = sweep_idle_instances(store, pod)

        assert stopped == 0
        pod.delete_pod.assert_not_called()

    def test_continues_sweep_when_one_pod_delete_fails(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_IDLE_TIMEOUT_HOURS", "2")
        old = datetime.now(timezone.utc) - timedelta(hours=3)
        inst_a = _make_instance(id="a", pod_name="pod-a", last_used_at=old)
        inst_b = _make_instance(id="b", pod_name="pod-b", last_used_at=old)

        store = _make_store([inst_a, inst_b])
        pod = _make_pod()
        pod.delete_pod.side_effect = [RuntimeError("K8s unreachable"), None]

        stopped = sweep_idle_instances(store, pod)

        # inst_a failed → marked FAILED; inst_b succeeded → marked SHUTDOWN
        assert stopped == 1
        assert store.update_status.call_count == 2
        store.update_status.assert_any_call("a", InferenceStatus.FAILED)
        store.update_status.assert_any_call("b", InferenceStatus.SHUTDOWN)

    def test_stops_multiple_idle_instances(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_IDLE_TIMEOUT_HOURS", "1")
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        instances = [
            _make_instance(id=f"inst-{i}", pod_name=f"pod-{i}", last_used_at=old)
            for i in range(3)
        ]
        store = _make_store(instances)
        pod = _make_pod()

        stopped = sweep_idle_instances(store, pod)

        assert stopped == 3
        assert pod.delete_pod.call_count == 3

    def test_respects_custom_timeout_env(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_IDLE_TIMEOUT_HOURS", "1")
        # idle for 90 minutes — past the 1h timeout
        marginally_idle = _make_instance(
            last_used_at=datetime.now(timezone.utc) - timedelta(minutes=90),
        )
        store = _make_store([marginally_idle])
        pod = _make_pod()

        stopped = sweep_idle_instances(store, pod)

        assert stopped == 1

    def test_invalid_timeout_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_IDLE_TIMEOUT_HOURS", "not-a-number")
        # Default is 2 hours; instance idle for 1h should NOT be stopped
        recent = _make_instance(
            last_used_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        store = _make_store([recent])
        pod = _make_pod()

        stopped = sweep_idle_instances(store, pod)

        assert stopped == 0


# ---------------------------------------------------------------------------
# check_initializing_instances — readiness watcher
# ---------------------------------------------------------------------------

class TestCheckInitializingInstances:
    from interactors.api.idle_shutdown import check_initializing_instances  # noqa: E402

    def _initializing(self, id: str = "inst-1") -> InferenceInstance:
        return _make_instance(id=id, status=InferenceStatus.INITIALIZING)

    def test_promotes_to_available_when_pod_running(self):
        from interactors.api.idle_shutdown import check_initializing_instances
        store = _make_store([self._initializing()])
        pod = _make_pod()
        pod.pod_status.return_value = "running"

        promoted = check_initializing_instances(store, pod)

        assert promoted == 1
        store.update_status.assert_called_once_with("inst-1", InferenceStatus.AVAILABLE)

    def test_marks_failed_when_pod_fails(self):
        from interactors.api.idle_shutdown import check_initializing_instances
        store = _make_store([self._initializing()])
        pod = _make_pod()
        pod.pod_status.return_value = "failed"

        promoted = check_initializing_instances(store, pod)

        assert promoted == 0
        store.update_status.assert_called_once_with("inst-1", InferenceStatus.FAILED)

    def test_skips_non_initializing_instances(self):
        from interactors.api.idle_shutdown import check_initializing_instances
        available = _make_instance(status=InferenceStatus.AVAILABLE)
        store = _make_store([available])
        pod = _make_pod()

        promoted = check_initializing_instances(store, pod)

        assert promoted == 0
        pod.pod_status.assert_not_called()

    def test_continues_on_pod_status_error(self):
        from interactors.api.idle_shutdown import check_initializing_instances
        store = _make_store([self._initializing()])
        pod = _make_pod()
        pod.pod_status.side_effect = Exception("k8s unavailable")

        # must not raise; instance stays untouched
        promoted = check_initializing_instances(store, pod)
        assert promoted == 0
        store.update_status.assert_not_called()

    def test_does_not_transition_on_unknown_phase(self):
        """Transient API errors ('unknown') must not change instance state."""
        from interactors.api.idle_shutdown import check_initializing_instances
        store = _make_store([self._initializing()])
        pod = _make_pod()
        pod.pod_status.return_value = "unknown"

        promoted = check_initializing_instances(store, pod)

        assert promoted == 0
        store.update_status.assert_not_called()

    def test_marks_failed_immediately_when_pod_not_found(self):
        """pod_status returns 'failed' for 404; polling loop must mark FAILED without delay."""
        from interactors.api.idle_shutdown import check_initializing_instances
        store = _make_store([self._initializing()])
        pod = _make_pod()
        # Adapter returns "failed" for 404 (pod deleted / never created)
        pod.pod_status.return_value = "failed"

        promoted = check_initializing_instances(store, pod)

        assert promoted == 0
        store.update_status.assert_called_once_with("inst-1", InferenceStatus.FAILED)
