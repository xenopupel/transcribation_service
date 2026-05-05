"""Dataclass schemas for the stereo ASR pipeline stages."""

from dataclasses import dataclass

import torch


@dataclass
class StereoSegment:
    """One VAD segment from stereo audio, before ASR inference."""
    job_id: str
    segment_id: int
    channel: int
    start: float
    end: float
    audio: torch.Tensor


@dataclass
class ASRSegmentResult:
    """ASR result for a single VAD segment."""
    job_id: str
    segment_id: int
    channel: int
    start: float
    end: float
    text: str
