"""Single-GPU background worker."""

from __future__ import annotations

import asyncio
import traceback

import torch

from gigaam_service.models import RuntimeModels, load_runtime_models
from gigaam_service.pipeline import transcribe_stereo_file

from .db import JobStore
from .settings import Settings
from .storage import delete_file_if_exists, save_result_files


class TranscriptionWorker:
    def __init__(self, *, queue: asyncio.Queue[str], store: JobStore, settings: Settings):
        self.queue = queue
        self.store = store
        self.settings = settings
        self.models: RuntimeModels | None = None
        self._stop = asyncio.Event()

    def load_models(self) -> None:
        self.models = load_runtime_models(
            model_name=self.settings.model_name,
            device=self.settings.device,
            load_spacy=self.settings.mask_pii,
            spacy_model_name=self.settings.spacy_model,
        )

    async def run(self) -> None:
        if self.models is None:
            self.load_models()

        while not self._stop.is_set():
            job_id = await self.queue.get()
            try:
                await asyncio.to_thread(self.process_job, job_id)
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        self._stop.set()

    def process_job(self, job_id: str) -> None:
        if self.models is None:
            raise RuntimeError("Models are not loaded")

        job = self.store.get_job(job_id)
        if job is None:
            return

        self.store.set_status(job_id, "processing")
        result_json_path = self.settings.results_dir / job_id / "result.json"
        result_txt_path = self.settings.results_dir / job_id / "result.txt"
        include_ivr = bool(job["include_ivr"])
        mask_pii = bool(job["mask_pii"])

        try:
            result = transcribe_stereo_file(
                audio_path=job["input_path"],
                model=self.models.asr_model,
                batch_size=self.settings.batch_size,
                pause_threshold=self.settings.pause_threshold,
                strict_limit_duration=self.settings.strict_limit_duration,
                operator_channel=self.settings.operator_channel,
                apply_postprocess=True,
                apply_masking=mask_pii,
                spacy_model=self.models.spacy_model if mask_pii else None,
                hold_threshold=self.settings.hold_threshold,
            )
            save_result_files(
                result,
                result_json_path,
                result_txt_path,
                include_ivr=include_ivr,
            )
            self.store.set_status(
                job_id,
                "done",
                result_json_path=result_json_path,
                result_txt_path=result_txt_path,
            )
        except RuntimeError as exc:
            error = str(exc)
            code = "gpu_oom" if "out of memory" in error.lower() else "runtime_error"
            self.store.set_status(
                job_id,
                "failed",
                error_code=code,
                error_message=error,
            )
        except Exception as exc:
            self.store.set_status(
                job_id,
                "failed",
                error_code=exc.__class__.__name__,
                error_message=f"{exc}\n{traceback.format_exc()}",
            )
        finally:
            delete_file_if_exists(job["input_path"])
            self._cleanup_cuda()

    @staticmethod
    def _cleanup_cuda() -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def enqueue_existing_jobs(queue: asyncio.Queue[str], store: JobStore) -> None:
    store.reset_processing_to_queued()
    for job in store.queued_jobs():
        queue.put_nowait(job["job_id"])
