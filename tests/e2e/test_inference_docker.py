"""Docker container smoke test — builds the inference image and verifies end-to-end inference.

The container downloads the GGUF from S3 on startup, so no volume mount is required.

Required env vars:
    INFERENCE_TEST_S3_GGUF   S3 key for a GGUF model (e.g. model/aipet.gguf)
    AWS_S3_BUCKET            S3 bucket name
    AWS_ACCESS_KEY_ID        AWS credentials
    AWS_SECRET_ACCESS_KEY    AWS credentials

Optional:
    AWS_SESSION_TOKEN        For temporary/STS credentials
    AWS_DEFAULT_REGION       Defaults to us-east-1

Run with:
    INFERENCE_TEST_S3_GGUF=model/aipet.gguf uv run pytest tests/e2e/test_inference_docker.py -v -s
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

_REQUIRED_VARS = [
    "INFERENCE_TEST_S3_GGUF",  # S3 key — container downloads this on startup
    "AWS_S3_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
]
_REPO_ROOT = Path(__file__).parents[2]
_IMAGE_TAG = "llm-api-inference-test:latest"
_STARTUP_TIMEOUT_S = 180  # allow time for S3 download + model load


def _missing_vars() -> list[str]:
    return [v for v in _REQUIRED_VARS if not os.environ.get(v)]


def _docker_available() -> bool:
    return shutil.which("docker") is not None


pytestmark = pytest.mark.skipif(
    not _docker_available() or bool(_missing_vars()),
    reason=(
        "docker not installed"
        if not _docker_available()
        else f"Missing env vars: {', '.join(_missing_vars())}"
    ),
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def inference_image() -> str:
    result = subprocess.run(
        ["docker", "build", "-f", "docker/inference/Dockerfile", "-t", _IMAGE_TAG, "."],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"docker build failed:\n{result.stderr}")
    return _IMAGE_TAG


@pytest.fixture(scope="session")
def inference_container(inference_image: str):
    """Start the inference container, let it download the GGUF from S3, yield base URL."""
    port = _free_port()

    passthrough_env = [
        "AWS_S3_BUCKET",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    ]
    env_args: list[str] = ["-e", f"GGUF_PATH={os.environ['INFERENCE_TEST_S3_GGUF']}"]
    for var in passthrough_env:
        if os.environ.get(var):
            env_args += ["-e", f"{var}={os.environ[var]}"]

    cmd = [
        "docker", "run", "-d",
        "-p", f"{port}:8080",
        *env_args,
        inference_image,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"docker run failed:\n{result.stderr}")

    container_id = result.stdout.strip()
    base_url = f"http://localhost:{port}"

    # Wait for /health — it returns 503 until the S3 download + model load completes
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    last_error = ""
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=3)
            if resp.status_code == 200:
                break
        except Exception as exc:
            last_error = str(exc)
        time.sleep(3)
    else:
        logs = subprocess.run(
            ["docker", "logs", container_id], capture_output=True, text=True
        ).stdout
        subprocess.run(["docker", "stop", container_id], capture_output=True)
        subprocess.run(["docker", "rm", container_id], capture_output=True)
        pytest.fail(
            f"Inference container never became healthy after {_STARTUP_TIMEOUT_S}s.\n"
            f"Last connection error: {last_error}\n"
            f"Container logs:\n{logs}"
        )

    yield base_url

    subprocess.run(["docker", "stop", container_id], capture_output=True)
    subprocess.run(["docker", "rm", container_id], capture_output=True)


# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

_PAYLOAD_HIGH_HUNGER = {
    "scene": {
        "objects": [
            {"id": "bowl1", "type": "bowl", "distance": 1.5},
            {"id": "toy1", "type": "toy", "distance": 3.0},
        ],
        "tick": 1,
    },
    "pet_stats": {
        "hunger": 0.9,
        "boredom": 0.1,
        "social": 0.1,
        "toilet": 0.1,
        "tiredness": 0.1,
    },
}

_PAYLOAD_EMPTY_SCENE = {
    "scene": {"objects": [], "tick": 1},
    "pet_stats": {
        "hunger": 0.5,
        "boredom": 0.3,
        "social": 0.2,
        "toilet": 0.1,
        "tiredness": 0.4,
    },
}

_VALID_ACTIONS = {"EAT", "DRINK", "PLAY", "FETCH", "SLEEP", "SOCIAL", "FOLLOW", "TOILET", "IDLE", "EXPLORE"}
_TARGET_REQUIRED_ACTIONS = {"EAT", "DRINK", "PLAY", "FETCH", "SLEEP", "SOCIAL", "FOLLOW"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInferenceDockerContainer:
    def test_health_returns_ready_with_model_loaded(self, inference_container: str) -> None:
        resp = httpx.get(f"{inference_container}/health", timeout=10)

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "ready", f"Unexpected health body: {body}"

    def test_infer_returns_http_200(self, inference_container: str) -> None:
        resp = httpx.post(f"{inference_container}/infer", json=_PAYLOAD_HIGH_HUNGER, timeout=60)

        assert resp.status_code == 200, f"/infer returned {resp.status_code}: {resp.text}"

    def test_infer_response_contains_valid_action(self, inference_container: str) -> None:
        resp = httpx.post(f"{inference_container}/infer", json=_PAYLOAD_HIGH_HUNGER, timeout=60)

        assert resp.status_code == 200
        action = resp.json()["action"]
        assert action in _VALID_ACTIONS, (
            f"Response action {action!r} is not a recognised Action value — "
            "parse_response may have silently returned a bad default"
        )

    def test_infer_does_not_return_503_due_to_libgomp_failure(self, inference_container: str) -> None:
        """The libgomp bug causes the model to never load, making health return 503
        and /infer return 503. A 200 from both endpoints proves libllama.so loaded
        successfully and inference actually ran."""
        health = httpx.get(f"{inference_container}/health", timeout=10)
        assert health.status_code == 200, (
            "Health check failed — model likely did not load. "
            "This is the libgomp symptom: check container logs for 'libgomp.so.1'."
        )

        resp = httpx.post(f"{inference_container}/infer", json=_PAYLOAD_HIGH_HUNGER, timeout=60)
        assert resp.status_code == 200, (
            f"/infer returned {resp.status_code} — model may not have loaded: {resp.text}"
        )

    def test_infer_empty_scene_returns_untargeted_action(self, inference_container: str) -> None:
        resp = httpx.post(f"{inference_container}/infer", json=_PAYLOAD_EMPTY_SCENE, timeout=60)

        assert resp.status_code == 200
        action = resp.json()["action"]
        assert action not in _TARGET_REQUIRED_ACTIONS, (
            f"Got {action} with an empty scene — a target-requiring action needs an object in the scene"
        )
