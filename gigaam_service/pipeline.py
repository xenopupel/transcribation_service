"""Synchronous stereo transcription pipeline."""

import subprocess
import uuid
from typing import Any, Dict, List, Optional

import torch

from . import CLIENT_CHANNEL, HOLD_CHANNEL, IVR_CHANNEL, OPERATOR_CHANNEL
from .asr_batch import transcribe_audio_batch
from .postprocess import apply_postprocessing
from .stereo_segmenter import segment_stereo_audio

# Default IVR detection triggers (from notebook, tuned for Hoff calls).
DEFAULT_START_TRIGGERS = [
    "разговор может быть записан",
    "дождитесь ответа оператора",
]
DEFAULT_END_TRIGGERS = [
    "оцените насколько оператор",
]


def _check_stereo(audio_path: str) -> None:
    """Raise ValueError if the audio file is not exactly 2 channels.

    Uses ffprobe (required by GigaAM anyway). If ffprobe is unavailable,
    the check is skipped and transcription will fail on its own if needed.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        num_channels = int(out.split()[0])
    except Exception:
        return  # ffprobe unavailable or parse error — skip check

    if num_channels != 2:
        raise ValueError(
            f"Only stereo (2-channel) audio is supported, "
            f"got {num_channels} channel(s): {audio_path}"
        )


@torch.inference_mode()
def _transcribe_batch(
    audio_path: str,
    job_id: str,
    model,
    batch_size: int,
    pause_threshold: float,
    strict_limit_duration: float,
) -> List[Dict[str, Any]]:
    """VAD + ASR via our own segmenter and batch function.

    Segments are sorted by audio length before batching (shorter padding),
    then results are restored to time order.
    """
    segments = segment_stereo_audio(
        audio_path=audio_path,
        job_id=job_id,
        pause_threshold=pause_threshold,
        strict_limit_duration=strict_limit_duration,
        device=model._device,
    )

    if not segments:
        return []

    # Sort by audio length (longest first) to minimise padding waste per batch.
    # Track original indices so we can restore time order after transcription.
    indexed = sorted(enumerate(segments), key=lambda x: len(x[1].audio), reverse=True)
    orig_indices = [i for i, _ in indexed]
    sorted_segs = [s for _, s in indexed]

    all_texts: List[str] = []
    for batch_start in range(0, len(sorted_segs), batch_size):
        batch = sorted_segs[batch_start:batch_start + batch_size]
        texts = transcribe_audio_batch(model, [s.audio for s in batch])
        all_texts.extend(texts)

    # Restore original (time) order
    texts_by_orig: List[str] = [""] * len(segments)
    for orig_idx, text in zip(orig_indices, all_texts):
        texts_by_orig[orig_idx] = text

    return [
        {
            "channel": seg.channel,
            "transcription": texts_by_orig[i],
            "boundaries": (seg.start, seg.end),
        }
        for i, seg in enumerate(segments)
    ]


def transcribe_stereo_file(
    audio_path: str,
    model,
    batch_size: int = 8,
    pause_threshold: float = 2.0,
    strict_limit_duration: float = 30.0,
    operator_channel: int = OPERATOR_CHANNEL,
    apply_postprocess: bool = True,
    apply_masking: bool = True,
    spacy_model=None,
    start_triggers: Optional[List[str]] = None,
    end_triggers: Optional[List[str]] = None,
    hold_threshold: float = 15.0,
) -> Dict[str, Any]:
    """Transcribe a single stereo audio file end-to-end.

    Steps:
      1. Validate stereo (exactly 2 channels).
      2. VAD + ASR via stereo_segmenter + asr_batch
         (segments sorted by length before batching to reduce padding).
      3. Optionally: relabel IVR, insert HOLD markers, mask PII.

    Returns a dict with job_id, audio_path, raw_segments, processed_segments,
    and channel ID constants.
    """
    _check_stereo(audio_path)

    job_id = str(uuid.uuid4())

    raw_segments = _transcribe_batch(
        audio_path, job_id, model, batch_size, pause_threshold, strict_limit_duration
    )

    raw_segments = sorted(raw_segments, key=lambda r: float(r["boundaries"][0]))

    processed_segments = [dict(r) for r in raw_segments]

    if apply_postprocess:
        _start = start_triggers if start_triggers is not None else DEFAULT_START_TRIGGERS
        _end = end_triggers if end_triggers is not None else DEFAULT_END_TRIGGERS

        processed_segments = apply_postprocessing(
            turns=processed_segments,
            operator_channel=operator_channel,
            start_triggers=_start,
            end_triggers=_end,
            ivr_channel=IVR_CHANNEL,
            hold_channel=HOLD_CHANNEL,
            hold_threshold=hold_threshold,
            mask_pii=apply_masking,
            spacy_model=spacy_model,
        )

    return {
        "job_id": job_id,
        "audio_path": audio_path,
        "raw_segments": raw_segments,
        "processed_segments": processed_segments,
        "operator_channel": operator_channel,
        "client_channel": CLIENT_CHANNEL,
        "hold_channel": HOLD_CHANNEL,
        "ivr_channel": IVR_CHANNEL,
    }
