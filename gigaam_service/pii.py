"""PII masking: phone numbers and NER-based personal data."""

import re
from typing import Any, Dict, List

_PHONE_RE = re.compile(
    r"(?:(?:\+7|7|8)\s*[-(]?\s*)?"
    r"(?:\d\s*[-)]?\s*){10,11}"
)


def mask_phone_numbers(text: str) -> str:
    """Replace Russian phone numbers with [MASKED_PHONE]."""
    if not text:
        return text
    return _PHONE_RE.sub("[MASKED_PHONE]", text)


def mask_personal_data(text: str, spacy_model=None) -> str:
    """Mask named entities with [MASKED_<LABEL>] and phone numbers.

    If spacy_model is None, only phone masking is applied.
    """
    if not text:
        return text

    masked = text
    if spacy_model is not None:
        doc = spacy_model(text)
        # Replace from end to keep offsets valid
        for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
            masked = masked[:ent.start_char] + f"[MASKED_{ent.label_}]" + masked[ent.end_char:]

    return mask_phone_numbers(masked)


def mask_clean_segments(
    turns: List[Dict[str, Any]],
    spacy_model=None,
    hold_channel: int = 2,
    ivr_channel: int = 3,
) -> List[Dict[str, Any]]:
    """Apply PII masking to operator/client segments. IVR and HOLD are left unchanged."""
    out: List[Dict[str, Any]] = []

    for item in turns:
        s = dict(item)
        ch = s.get("channel")
        txt = s.get("transcription") or ""

        if ch not in (hold_channel, ivr_channel) and txt.strip():
            s["transcription"] = mask_personal_data(txt, spacy_model)

        out.append(s)

    return out
