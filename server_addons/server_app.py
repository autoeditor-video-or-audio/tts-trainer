"""FastAPI app for tts-trainer."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, RedirectResponse

from server_addons import datasets, training, voicepack
from server_addons.schemas import (
    DatasetDownloadRequest,
    DatasetItem,
    DatasetList,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    TrainingRunCreate,
    TrainingRunCreateResponse,
    TrainingRunDetail,
    TrainingRunList,
    VoicepackExportRequest,
    VoicepackItem,
    VoicepackList,
)
from server_addons.storage import get_storage
from server_addons.voicepack import ShapeMismatch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("TTS_TRAINER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    storage = get_storage()
    storage.ensure_bucket()
    # Warm the executor lazily; importing training already created it.
    training.executor()
    yield
    if training._executor is not None:
        training._executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="tts-trainer (autoeditor fork)", version="0.1.0", lifespan=lifespan)


def _err(message: str, status: int = 400, actual_shape: list | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(message=message, actual_shape=actual_shape)).model_dump(
        exclude_none=True
    )
    return JSONResponse(status_code=status, content=body)


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health() -> HealthResponse:
    storage_ok = get_storage().health()
    cuda_ok = _cuda_available()
    return HealthResponse(
        ready=storage_ok and cuda_ok,
        storage_reachable=storage_ok,
        cuda_available=cuda_ok,
    )


@app.get("/v1/datasets", response_model=DatasetList)
async def list_datasets() -> DatasetList:
    items = [
        DatasetItem(
            id=d.id,
            hours=d.hours,
            speakers=d.speakers,
            downloaded_at=d.downloaded_at,
            status=d.status,
        )
        for d in datasets.list_local()
    ]
    return DatasetList(datasets=items)


@app.post("/v1/datasets/download", status_code=202)
async def download_dataset(req: DatasetDownloadRequest) -> JSONResponse:
    loop = asyncio.get_running_loop()
    # Run the (potentially long) download in a thread so the request
    # returns 202 immediately.
    async def _run() -> None:
        try:
            await loop.run_in_executor(None, datasets.download_dataset, req.id)
        except Exception:
            logger.exception("dataset %s download failed", req.id)

    asyncio.create_task(_run())
    return JSONResponse(status_code=202, content={"dataset_id": req.id, "status": "downloading"})


@app.post("/v1/training/runs", response_model=TrainingRunCreateResponse)
async def create_run(req: TrainingRunCreate) -> TrainingRunCreateResponse:
    if not datasets.is_ready(req.dataset_id):
        raise HTTPException(status_code=400, detail=f"dataset {req.dataset_id!r} not ready")
    run_id = training.queue_run(
        dataset_id=req.dataset_id,
        name=req.name,
        voicepack_name=req.voicepack_name,
        language=req.language,
        epochs=req.epochs,
        lr=req.lr,
    )
    return TrainingRunCreateResponse(run_id=run_id, status="queued")


@app.get("/v1/training/runs", response_model=TrainingRunList)
async def list_runs() -> TrainingRunList:
    runs = [
        TrainingRunDetail(
            run_id=r["run_id"],
            name=r["name"],
            dataset_id=r["dataset_id"],
            voicepack_name=r["voicepack_name"],
            language=r["language"],
            status=r["status"],
            epoch=r.get("epoch"),
            loss=r.get("loss"),
            eta_seconds=r.get("eta_seconds"),
            started_at=r.get("started_at"),
            finished_at=r.get("finished_at"),
            log_tail=None,  # heavy; only surface on detail endpoint
        )
        for r in training.list_runs()
    ]
    return TrainingRunList(runs=runs)


@app.get("/v1/training/runs/{run_id}", response_model=TrainingRunDetail)
async def get_run(run_id: str) -> TrainingRunDetail:
    row = training.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return TrainingRunDetail(
        run_id=row["run_id"],
        name=row["name"],
        dataset_id=row["dataset_id"],
        voicepack_name=row["voicepack_name"],
        language=row["language"],
        status=row["status"],
        epoch=row.get("epoch"),
        loss=row.get("loss"),
        eta_seconds=row.get("eta_seconds"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        log_tail=row.get("log_tail"),
    )


@app.delete("/v1/training/runs/{run_id}", status_code=204)
async def cancel_run(run_id: str) -> Response:
    if not training.cancel_run(run_id):
        raise HTTPException(status_code=404, detail="run not found or not cancellable")
    return Response(status_code=204)


@app.post("/v1/voicepacks/{run_id}/export", response_model=VoicepackItem)
async def export_voicepack(run_id: str, req: VoicepackExportRequest) -> VoicepackItem:
    run = training.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run["status"] != "ready":
        raise HTTPException(status_code=400, detail=f"run status is {run['status']!r}, expected 'ready'")
    loop = asyncio.get_running_loop()
    try:
        info = await loop.run_in_executor(None, voicepack.export_voicepack, run_id, req.name)
    except ShapeMismatch as e:
        return _err(str(e), status=400, actual_shape=list(e.actual_shape))  # type: ignore[return-value]
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return VoicepackItem(**info)


@app.get("/v1/voicepacks", response_model=VoicepackList)
async def list_voicepacks() -> VoicepackList:
    return VoicepackList(voicepacks=[VoicepackItem(**v) for v in voicepack.list_voicepacks()])


@app.get("/v1/voicepacks/{name}/download")
async def download_voicepack(name: str) -> Response:
    url = voicepack.presign_voicepack(name)
    if url is None:
        raise HTTPException(status_code=404, detail="voicepack not found")
    return RedirectResponse(url=url, status_code=302)
