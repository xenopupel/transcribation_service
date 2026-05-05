"""File storage helpers for uploaded audio and transcription results."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from fastapi import HTTPException, UploadFile, status

from gigaam_service.formatter import save_transcript_txt

from .settings import Settings


AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def init_storage(settings: Settings) -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)


def delete_file_if_exists(path: Path | str | None) -> None:
    if path is None:
        return
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def ensure_disk_space(settings: Settings) -> None:
    usage = shutil.disk_usage(settings.storage_dir)
    if usage.free < settings.min_free_disk_bytes:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Not enough free disk space for new jobs",
        )


def validate_audio_filename(filename: str) -> None:
    if Path(filename).suffix.lower() not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio extension. Supported: {sorted(AUDIO_EXTENSIONS)}",
        )


async def save_upload_file(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Uploaded file exceeds configured size limit",
                )
            f.write(chunk)
    await upload.close()
    return total


def serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    for key in ("raw_segments", "processed_segments"):
        segments = []
        for seg in out.get(key, []):
            item = dict(seg)
            if isinstance(item.get("boundaries"), tuple):
                item["boundaries"] = list(item["boundaries"])
            segments.append(item)
        out[key] = segments
    return out


def filter_result_for_output(result: dict[str, Any], *, include_ivr: bool) -> dict[str, Any]:
    if include_ivr:
        return result

    out = dict(result)
    ivr_channel = out["ivr_channel"]
    out["processed_segments"] = [
        dict(seg)
        for seg in out.get("processed_segments", [])
        if seg.get("channel") != ivr_channel
    ]
    return out


def save_result_files(
    result: dict[str, Any],
    result_json_path: Path,
    result_txt_path: Path,
    *,
    include_ivr: bool = True,
) -> None:
    output_result = filter_result_for_output(result, include_ivr=include_ivr)
    serialized = serialize_result(output_result)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)

    save_transcript_txt(
        turns=output_result["processed_segments"],
        path=str(result_txt_path),
        operator_channel=result["operator_channel"],
        hold_channel=result["hold_channel"],
        ivr_channel=result["ivr_channel"],
        include_ivr=include_ivr,
    )
