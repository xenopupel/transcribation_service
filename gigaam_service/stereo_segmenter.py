"""VAD segmentation for stereo audio files.

Returns StereoSegment objects (audio on device, no transcription yet).
Delegates to GigaAM's segment_multichannel_audio — does not duplicate VAD logic.
"""

import torch
from gigaam.preprocess import SAMPLE_RATE
from gigaam.vad_utils import segment_multichannel_audio

from .schemas import StereoSegment


def segment_stereo_audio(
    audio_path: str,
    job_id: str,
    sample_rate: int = SAMPLE_RATE,
    pause_threshold: float = 2.0,
    strict_limit_duration: float = 30.0,
    device: torch.device = torch.device("cpu"),
) -> list[StereoSegment]:
    """Run VAD on a stereo audio file and return segments ready for ASR.

    Audio tensors in returned segments are already on `device`.
    Segments are sorted by start time and assigned sequential segment_ids.
    """
    raw = segment_multichannel_audio(
        audio_input=audio_path,
        sr=sample_rate,
        pause_threshold=pause_threshold,
        strict_limit_duration=strict_limit_duration,
        device=device,
    )

    raw = sorted(raw, key=lambda s: s["boundaries"][0])

    return [
        StereoSegment(
            job_id=job_id,
            segment_id=idx,
            channel=int(seg["channel"]),
            start=float(seg["boundaries"][0]),
            end=float(seg["boundaries"][1]),
            audio=seg["audio"],
        )
        for idx, seg in enumerate(raw)
    ]
