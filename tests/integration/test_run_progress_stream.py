"""Integration tests for GET /api/runs/{run_id}/progress/stream."""
import json
import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport

from domain.models import RunRecord, RunStatus


def make_run(run_id: str, status: RunStatus, progress: float | None, detail: str | None = None):
    return RunRecord(
        id=run_id,
        model_id="model-1",
        workflow_id="wf-1",
        status=status,
        progress=progress,
        progress_detail=detail,
        owner_id=None,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )


@pytest.fixture()
def app_with_overrides():
    """Return the FastAPI app with auth and run_store overridden."""
    from interactors.api.app import app
    from interactors.api.deps import get_run_store
    from interactors.api.auth import require_approved
    from domain.models import UserContext

    app.dependency_overrides[require_approved] = lambda: UserContext(
        user_id="user-1", email="t@t.com", approved=True
    )
    yield app
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_progress_stream_404_for_unknown_run(app_with_overrides):
    from interactors.api.deps import get_run_store
    run_store = MagicMock()
    run_store.get.return_value = None
    app_with_overrides.dependency_overrides[get_run_store] = lambda: run_store

    async with AsyncClient(transport=ASGITransport(app=app_with_overrides), base_url="http://test") as client:
        response = await client.get("/api/runs/unknown-id/progress/stream")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_progress_stream_emits_event_then_closes_on_terminal(app_with_overrides):
    """Stream emits a progress event then sends event:done when run becomes terminal."""
    from interactors.api.deps import get_run_store
    run_id = "run-progress-test"
    call_count = 0

    def get_side_effect(id):
        nonlocal call_count
        call_count += 1
        # call 1 = upfront auth check in endpoint; calls 2+ = inside generator loop
        if call_count <= 2:
            return make_run(id, RunStatus.TRAINING, 0.5, "step 50/100")
        return make_run(id, RunStatus.COMPLETED, 1.0, "done")

    run_store = MagicMock()
    run_store.get.side_effect = get_side_effect
    app_with_overrides.dependency_overrides[get_run_store] = lambda: run_store

    collected: list[str] = []
    async with AsyncClient(transport=ASGITransport(app=app_with_overrides), base_url="http://test") as client:
        async with client.stream("GET", f"/api/runs/{run_id}/progress/stream") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            async for line in response.aiter_lines():
                collected.append(line)
                if "event: done" in line or len(collected) > 30:
                    break

    data_lines = [l for l in collected if l.startswith("data:") and "stream closed" not in l]
    assert len(data_lines) >= 1, f"expected at least one data event; got: {collected}"
    payload = json.loads(data_lines[0].removeprefix("data: "))
    assert payload["fraction"] == 0.5
    assert payload["detail"] == "step 50/100"
    assert any("event: done" in l for l in collected), f"expected event:done; got: {collected}"
