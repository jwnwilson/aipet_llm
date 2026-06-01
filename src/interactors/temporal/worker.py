"""Temporal worker — registers all activities and the workflow, then polls for tasks."""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from temporalio.client import Client
from temporalio.worker import Worker

from interactors.temporal.activities import (
    configure_inference_store,
    configure_model_store,
    configure_run_store,
    configure_storage,
    create_inference_activity,
    evaluate_activity,
    export_activity,
    fail_run_activity,
    finalise_run_activity,
    generate_dataset_activity,
    record_eval_result_activity,
    save_gguf_path_activity,
    train_activity,
    update_run_status_activity,
)
from interactors.temporal.workflows import EvaluateWorkflow, ExportWorkflow, TrainingPipelineWorkflow

TASK_QUEUE = "llm-api-training"


async def main() -> None:
    from adapters.database import init_db, make_engine
    from adapters.database.model_store import SQLAlchemyModelStore
    from adapters.database.run_store import SQLAlchemyRunStore

    engine = make_engine()
    init_db(engine)
    from adapters.database.inference_store import SQLAlchemyInferenceStore
    configure_model_store(SQLAlchemyModelStore(engine))
    configure_run_store(SQLAlchemyRunStore(engine))
    configure_inference_store(SQLAlchemyInferenceStore(engine))

    if os.getenv("AWS_S3_BUCKET"):
        from adapters.storage.s3 import S3StorageAdapter
        configure_storage(S3StorageAdapter())
    else:
        from adapters.storage.local import LocalStorageAdapter
        configure_storage(LocalStorageAdapter())

    temporal_host = os.environ.get("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)

    worker_count = int(os.environ.get("TEMPORAL_WORKER_COUNT", "2"))
    worker_kwargs = dict(
        task_queue=TASK_QUEUE,
        workflows=[TrainingPipelineWorkflow, EvaluateWorkflow, ExportWorkflow],
        activities=[
            generate_dataset_activity,
            train_activity,
            evaluate_activity,
            export_activity,
            fail_run_activity,
            finalise_run_activity,
            record_eval_result_activity,
            save_gguf_path_activity,
            update_run_status_activity,
            create_inference_activity,
        ],
    )
    workers = [Worker(client, **worker_kwargs) for _ in range(worker_count)]

    log = logging.getLogger(__name__)
    log.info("Starting %d worker(s) — task_queue=%s  host=%s", worker_count, TASK_QUEUE, temporal_host)
    await asyncio.gather(*[w.run() for w in workers])


if __name__ == "__main__":
    asyncio.run(main())
