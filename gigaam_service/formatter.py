"""Output formatting: save transcript to plain-text file."""

from typing import Any, Dict, List


def save_transcript_txt(
    turns: List[Dict[str, Any]],
    path: str,
    operator_channel: int = 1,
    hold_channel: int = 2,
    ivr_channel: int = 3,
    include_ivr: bool = True,
) -> None:
    """Write transcript to a UTF-8 text file.

    Each turn on its own paragraph, double-newline separated:
        IVR: текст          (only when include_ivr=True)
        Оператор: текст
        Клиент: текст
        [CALL_ON_HOLD]
    """
    lines = []

    for r in turns:
        text = (r.get("transcription") or "").strip()
        if not text:
            continue

        channel = r.get("channel")

        if channel == ivr_channel:
            if include_ivr:
                lines.append(f"IVR: {text}")
        elif channel == hold_channel:
            lines.append("[CALL_ON_HOLD]")
        else:
            speaker = "Оператор" if channel == operator_channel else "Клиент"
            lines.append(f"{speaker}: {text}")

    # Drop leading HOLD markers (they appear at the start when IVR is excluded).
    while lines and lines[0] == "[CALL_ON_HOLD]":
        lines.pop(0)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
