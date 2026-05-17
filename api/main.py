"""FastAPI entrypoint for the transcription service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .db import JobStore
from .schemas import JobCreated, JobInfo, JobResult, JobsStartRequest, JobsStartResponse
from .settings import settings
from .storage import (
    AUDIO_EXTENSIONS,
    delete_file_if_exists,
    ensure_disk_space,
    init_storage,
    save_upload_file,
    serialize_result,
    validate_audio_filename,
)
from .worker import TranscriptionWorker, enqueue_existing_jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_storage(settings)
    store = JobStore(settings.db_path)
    for job in store.terminal_jobs():
        delete_file_if_exists(job["input_path"])

    queue: asyncio.Queue[str] = asyncio.Queue()
    worker = TranscriptionWorker(queue=queue, store=store, settings=settings)

    await asyncio.to_thread(worker.load_models)
    enqueue_existing_jobs(queue, store)
    worker_task = asyncio.create_task(worker.run())

    app.state.store = store
    app.state.queue = queue
    app.state.worker = worker
    app.state.worker_task = worker_task

    try:
        yield
    finally:
        worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Transcribe Service", version="0.1.0", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _job_info(job: dict) -> JobInfo:
    job_id = job["job_id"]
    done = job["status"] == "done"
    return JobInfo(
        job_id=job_id,
        status=job["status"],
        filename=job["filename"],
        include_ivr=bool(job["include_ivr"]),
        mask_pii=bool(job["mask_pii"]),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        queue_position=app.state.store.queue_position(job_id),
        error_code=job["error_code"],
        error_message=job["error_message"],
        result_json_url=f"/jobs/{job_id}/result.json" if done else None,
        result_txt_url=f"/jobs/{job_id}/result.txt" if done else None,
    )


def _get_job_or_404(job_id: str) -> dict:
    job = app.state.store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


def _txt_archive_name(filename: str, used_names: set[str]) -> str:
    base = Path(filename).stem or "transcript"
    candidate = f"{base}.txt"
    index = 2
    while candidate.lower() in used_names:
        candidate = f"{base} ({index}).txt"
        index += 1
    used_names.add(candidate.lower())
    return candidate


@app.get("/")
def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, object]:
    return {
        "audio_extensions": sorted(AUDIO_EXTENSIONS),
        "max_upload_mb": settings.max_upload_mb,
        "max_queued_jobs": settings.max_queued_jobs,
        "default_include_ivr": False,
        "mask_pii": settings.mask_pii,
    }


@app.post("/jobs", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    file: UploadFile = File(...),
    include_ivr: bool = Query(False, description="Include IVR segments in result JSON/TXT"),
    enqueue: bool = Query(True, description="Put uploaded file into processing queue"),
) -> JobCreated:
    store: JobStore = app.state.store
    queue: asyncio.Queue[str] = app.state.queue

    if enqueue and store.count_active_jobs() >= settings.max_queued_jobs:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many queued or processing jobs",
        )

    filename = Path(file.filename or "audio").name
    validate_audio_filename(filename)
    ensure_disk_space(settings)

    job_id = uuid4().hex
    input_path = settings.uploads_dir / f"{job_id}{Path(filename).suffix.lower()}"
    await save_upload_file(file, input_path, settings.max_upload_bytes)

    store.create_job(
        job_id,
        filename,
        input_path,
        include_ivr=include_ivr,
        mask_pii=settings.mask_pii,
        status="queued" if enqueue else "uploaded",
    )
    if enqueue:
        await queue.put(job_id)

    return JobCreated(job_id=job_id, status="queued" if enqueue else "uploaded")


@app.post("/jobs/start", response_model=JobsStartResponse)
async def start_jobs(payload: JobsStartRequest) -> JobsStartResponse:
    store: JobStore = app.state.store
    queue: asyncio.Queue[str] = app.state.queue

    if not payload.job_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No job IDs provided",
        )
    if store.count_active_jobs() + len(payload.job_ids) > settings.max_queued_jobs:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many queued or processing jobs",
        )

    started = store.mark_uploaded_as_queued(payload.job_ids)
    for job_id in started:
        await queue.put(job_id)
    return JobsStartResponse(started=len(started))


@app.get("/jobs/archive")
def download_jobs_archive(job_id: list[str] = Query(...)) -> Response:
    archive = BytesIO()
    used_names: set[str] = set()
    added = 0

    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
        for item in job_id:
            job = _get_job_or_404(item)
            if job["status"] == "failed":
                continue
            if job["status"] != "done":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Job is not done. Current status: {job['status']}",
                )

            txt_path = Path(job["result_txt_path"])
            if not txt_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Text result file not found",
                )

            archive_name = _txt_archive_name(job["filename"], used_names)
            zip_file.write(txt_path, archive_name)
            added += 1

    if added == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No completed text results found for archive",
        )

    archive.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="transcripts.zip"'}
    return Response(content=archive.getvalue(), media_type="application/zip", headers=headers)


@app.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str) -> JobInfo:
    return _job_info(_get_job_or_404(job_id))


@app.get("/jobs/{job_id}/result", response_model=JobResult)
def get_job_result(job_id: str) -> JobResult:
    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not done. Current status: {job['status']}",
        )

    result_path = Path(job["result_json_path"])
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return JobResult(job_id=job_id, result=serialize_result(result))


@app.get("/jobs/{job_id}/result.json")
def download_result_json(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is not done")
    return FileResponse(Path(job["result_json_path"]), media_type="application/json")


@app.get("/jobs/{job_id}/result.txt")
def download_result_txt(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is not done")
    return FileResponse(Path(job["result_txt_path"]), media_type="text/plain; charset=utf-8")
