from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


RunStatus = Literal["queued", "running", "ready", "failed"]
DatasetStatus = Literal["downloading", "ready", "failed"]


class DatasetDownloadRequest(BaseModel):
    id: str = Field(description="Known dataset id (e.g. 'commonvoice_ptbr', 'cetuc', 'coraa') or 'hf:<owner>/<repo>'.")


class DatasetItem(BaseModel):
    id: str
    hours: Optional[float] = None
    speakers: Optional[int] = None
    downloaded_at: Optional[str] = None
    status: DatasetStatus


class DatasetList(BaseModel):
    datasets: list[DatasetItem]


class TrainingRunCreate(BaseModel):
    dataset_id: str
    name: str = Field(description="Human-readable run name")
    language: str = Field(default="pt-BR")
    voicepack_name: str = Field(description="File stem for the exported voicepack (no .pt)")
    epochs: Optional[int] = Field(default=None, ge=1, le=400)
    lr: Optional[float] = Field(default=None, gt=0)


class TrainingRunCreateResponse(BaseModel):
    run_id: str
    status: RunStatus


class TrainingRunDetail(BaseModel):
    run_id: str
    name: str
    dataset_id: str
    voicepack_name: str
    language: str
    status: RunStatus
    epoch: Optional[int] = None
    loss: Optional[float] = None
    eta_seconds: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log_tail: Optional[str] = None


class TrainingRunList(BaseModel):
    runs: list[TrainingRunDetail]


class VoicepackExportRequest(BaseModel):
    name: str


class VoicepackItem(BaseModel):
    name: str
    size_bytes: int
    created_at: Optional[str] = None
    run_id: Optional[str] = None
    sha256: Optional[str] = None


class VoicepackList(BaseModel):
    voicepacks: list[VoicepackItem]


class HealthResponse(BaseModel):
    ready: bool
    storage_reachable: bool
    cuda_available: bool


class ErrorBody(BaseModel):
    message: str
    actual_shape: Optional[list[int]] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
