"""API response schemas."""

from typing import Any, Literal

from pydantic import BaseModel


JobStatus = Literal["uploaded", "queued", "processing", "done", "failed", "rejected"]


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus


class JobsStartRequest(BaseModel):
    job_ids: list[str]


class JobsStartResponse(BaseModel):
    started: int


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    include_ivr: bool
    mask_pii: bool
    created_at: str
    updated_at: str
    queue_position: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    result_json_url: str | None = None
    result_txt_url: str | None = None


class JobResult(BaseModel):
    job_id: str
    result: dict[str, Any]
