"""Runtime model loading for ASR, VAD and PII postprocessing."""

from dataclasses import dataclass
from typing import Any

import gigaam
from gigaam.vad_utils import get_pipeline
import spacy


@dataclass
class RuntimeModels:
    asr_model: Any
    spacy_model: Any | None = None


def load_runtime_models(
    *,
    model_name: str = "v3_e2e_rnnt",
    device: str | None = "cuda",
    load_spacy: bool = True,
    spacy_model_name: str = "ru_core_news_lg",
) -> RuntimeModels:
    """Load all long-lived models used by the transcription pipeline."""
    asr_model = gigaam.load_model(model_name, device=device)

    # Warm up and cache pyannote VAD pipeline inside GigaAM's vad_utils module.
    get_pipeline(asr_model._device)

    spacy_model = spacy.load(spacy_model_name) if load_spacy else None
    return RuntimeModels(asr_model=asr_model, spacy_model=spacy_model)
