"""FastAPI application factory for the llm-api inference service."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)


def _make_storage_adapter():
    bucket = os.getenv("AWS_S3_BUCKET")
    if bucket:
        log.info("Storage: S3StorageAdapter  bucket=%s", bucket)
        from adapters.storage.s3 import S3StorageAdapter
        return S3StorageAdapter()
    log.warning(
        "Storage: LocalStorageAdapter (AWS_S3_BUCKET not set) — "
        "dataset uploads will be stored on local disk only. "
        "Remote training backends (RunPod, VastAI, K8s) will fail to download datasets."
    )
    from adapters.storage.local import LocalStorageAdapter
    return LocalStorageAdapter()


def _resolve_model_path(storage) -> str:
    """Return a local model path, downloading from S3 via MODEL_S3_KEY if configured."""
    from adapters.storage import download_model
    s3_key = os.getenv("MODEL_S3_KEY")
    if s3_key:
        local_path = Path("models/cache/default/model.gguf")
        try:
            download_model(storage, s3_key, local_path)
            log.info("Downloaded model from MODEL_S3_KEY=%s to %s", s3_key, local_path)
            return str(local_path)
        except Exception:
            log.warning("Could not download model from MODEL_S3_KEY=%s", s3_key, exc_info=True)
    return os.getenv("MODEL_PATH", "models/model.gguf")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from adapters.auth.auth0 import Auth0Adapter
    from adapters.database import init_db, make_engine
    from adapters.inference import LlamaCppInferenceAdapter
    from interactors.api.deps import (
        clear_adapter,
        clear_auth,
        clear_storage,
        clear_uow,
        configure,
        configure_auth,
        configure_storage as configure_api_storage,
        configure_uow,
    )
    from interactors.temporal.activities import (
        configure_engine as configure_activity_engine,
        configure_storage,
    )

    engine = make_engine()
    init_db(engine)

    configure_uow(engine)
    configure_activity_engine(engine)

    storage = _make_storage_adapter()
    configure_storage(storage)
    configure_api_storage(storage)

    auth_disabled = os.getenv("AUTH_DISABLED", "").lower() == "true"
    auth0_domain = os.environ.get("AUTH0_DOMAIN", "")
    auth0_audience = os.environ.get("AUTH0_AUDIENCE", "")
    if auth_disabled:
        from adapters.auth.fake import FakeAuthAdapter
        log.warning("AUTH_DISABLED=true — using FakeAuthAdapter, all requests treated as admin")
        configure_auth(FakeAuthAdapter())
    elif auth0_domain and auth0_audience:
        configure_auth(Auth0Adapter(domain=auth0_domain, audience=auth0_audience))
    elif os.getenv("APP_ENV") == "development":
        from adapters.auth.fake import FakeAuthAdapter
        log.warning("AUTH0 not configured — using FakeAuthAdapter for local development")
        configure_auth(FakeAuthAdapter())
    else:
        log.warning(
            "AUTH0_DOMAIN or AUTH0_AUDIENCE not set — "
            "protected endpoints will return 500 until configured"
        )

    inference_disabled = os.getenv("INFERENCE_DISABLED", "false").lower() == "true"

    if inference_disabled:
        log.info("INFERENCE_DISABLED=true — skipping local model download and inference adapter")
    else:
        from adapters.storage import download_model
        from adapters.database.uow import SQLAlchemyUnitOfWork
        with SQLAlchemyUnitOfWork(engine).transaction() as _startup_uow:
            active = _startup_uow.model_store.active()
        if active and active.gguf_path:
            local_path = Path("models/cache") / active.id / "model.gguf"
            try:
                download_model(storage, active.gguf_path, local_path)
                model_path = str(local_path)
                log.info("Loading active model %s from storage key %s", active.id, active.gguf_path)
            except Exception:
                log.warning(
                    "Could not load active model %s from storage; falling back",
                    active.id,
                    exc_info=True,
                )
                model_path = _resolve_model_path(storage)
        else:
            model_path = _resolve_model_path(storage)

        adapter = LlamaCppInferenceAdapter(model_path=model_path)
        configure(adapter)
        log.info("Inference adapter configured (model will load on first request): %s", model_path)

    from adapters.compute.k8s.adapter import K8sPodAdapter, MockPodAdapter
    from interactors.api.deps import clear_pod_adapter, configure_pod_adapter
    if os.environ.get("K8S_MOCK", "false").lower() == "true":
        pod_adapter = MockPodAdapter()
        configure_pod_adapter(pod_adapter)
        log.info("Pod adapter: MockPodAdapter (K8S_MOCK=true)")
    else:
        pod_adapter = K8sPodAdapter()
        configure_pod_adapter(pod_adapter)
        log.info("Pod adapter: K8sPodAdapter")

    from adapters.database.inference_store import SQLAlchemyInferenceStore
    from interactors.api.idle_shutdown import idle_shutdown_loop, readiness_watch_loop
    from sqlalchemy.orm import Session as _Session

    def _bg_store_factory() -> SQLAlchemyInferenceStore:
        return SQLAlchemyInferenceStore(_Session(engine))

    shutdown_task = asyncio.create_task(idle_shutdown_loop(_bg_store_factory, pod_adapter))
    readiness_task = asyncio.create_task(readiness_watch_loop(_bg_store_factory, pod_adapter))
    log.info("Idle inference shutdown task started")
    log.info("Inference readiness watcher task started")

    try:
        yield
    finally:
        shutdown_task.cancel()
        readiness_task.cancel()
        clear_adapter()
        clear_auth()
        clear_uow()
        clear_storage()
        clear_pod_adapter()


from interactors.api.routes.admin import router as admin_router  # noqa: E402
from interactors.api.routes.datasets import router as datasets_router  # noqa: E402
from interactors.api.routes.inference import router as inference_router  # noqa: E402
from interactors.api.routes.inferences import router as inferences_router  # noqa: E402
from interactors.api.routes.login import router as login_router  # noqa: E402
from interactors.api.routes.models import router as models_router  # noqa: E402
from interactors.api.routes.runs import router as runs_router  # noqa: E402

_auth0_audience = os.getenv("AUTH0_AUDIENCE", "")
_auth0_client_id = os.getenv("AUTH0_CLIENT_ID", "")

app = FastAPI(
    title="llm-api inference service",
    lifespan=lifespan,
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
    swagger_ui_init_oauth={
        "clientId": _auth0_client_id,
        "additionalQueryStringParams": {"audience": _auth0_audience},
        "usePkceWithAuthorizationCodeGrant": True,
        "scopes": "openid profile email",
    },
)

_cors_raw = os.getenv("CORS_ORIGINS", "")
if os.getenv("APP_ENV") == "development":
    _cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]
elif _cors_raw:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    _cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(admin_router)
app.include_router(datasets_router)
app.include_router(inference_router)
app.include_router(inferences_router)
app.include_router(models_router)
app.include_router(runs_router)
app.include_router(login_router)

