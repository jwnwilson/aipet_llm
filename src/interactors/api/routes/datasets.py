"""Dataset file upload endpoints."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from domain.ports import StoragePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_storage

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/datasets",
    tags=["datasets"],
    dependencies=[Depends(require_approved)],
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


class DatasetUploadResult(BaseModel):
    key: str


CHUNK_SIZE = 64 * 1024  # 64 KB read buffer


async def _upload_dataset(file: UploadFile, storage: StoragePort, key: str) -> dict[str, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds maximum upload size (50 MB)")
            tmp.write(chunk)

    if size == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        storage.upload(tmp_path, key)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("Uploaded dataset: key=%s bytes=%d", key, size)
    return {"key": key}


@router.post("/train", status_code=201, response_model=DatasetUploadResult)
async def upload_train_dataset(
    file: UploadFile = File(...),
    storage: StoragePort = Depends(get_storage),
) -> DatasetUploadResult:
    return DatasetUploadResult(**await _upload_dataset(file, storage, "datasets/train.jsonl"))


@router.post("/eval", status_code=201, response_model=DatasetUploadResult)
async def upload_eval_dataset(
    file: UploadFile = File(...),
    storage: StoragePort = Depends(get_storage),
) -> DatasetUploadResult:
    return DatasetUploadResult(**await _upload_dataset(file, storage, "datasets/eval.jsonl"))
