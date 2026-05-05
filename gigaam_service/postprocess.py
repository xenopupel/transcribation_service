"""Stereo-specific postprocessing: IVR relabeling, HOLD insertion and PII masking."""

import bisect
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .pii import mask_clean_segments

_LEADING_JUNK = re.compile(r'^[\s\.]+')
_word_re = re.compile(r"[0-9a-zа-яё]+", flags=re.IGNORECASE)


def _clean_leading_junk(s: str) -> str:
    return _LEADING_JUNK.sub("", s or "")


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """Lowercase, ё→е, keep only alnum tokens, join by spaces.

    Returns (norm_text, norm2orig) where norm2orig[k] is the original char index.
    """
    if not text:
        return "", []

    lower = text.lower().replace("ё", "е")
    tokens = list(_word_re.finditer(lower))

    norm_chars: List[str] = []
    norm2orig: List[int] = []

    first = True
    for m in tokens:
        if not first:
            norm_chars.append(" ")
            norm2orig.append(m.start())
        first = False
        for j, ch in enumerate(lower[m.start():m.end()]):
            norm_chars.append(ch)
            norm2orig.append(m.start() + j)

    return "".join(norm_chars), norm2orig


@dataclass
class _OpUtter:
    idx_in_results: int
    norm: str
    norm2orig: List[int]
    global_start: int
    global_end: int  # exclusive


def _build_operator_stream(
    results: List[Dict[str, Any]], operator_channel: int
) -> Tuple[str, List[_OpUtter]]:
    """Concatenate all operator utterances into one normalized stream with position metadata."""
    op_utts: List[_OpUtter] = []
    parts: List[str] = []
    pos = 0

    for i, r in enumerate(results):
        if r.get("channel") != operator_channel:
            continue
        txt = (r.get("transcription") or "")
        if not txt.strip():
            continue

        norm, norm2orig = _normalize_with_map(txt)
        if not norm:
            continue

        if parts:
            parts.append(" ")
            pos += 1

        global_start = pos
        parts.append(norm)
        pos += len(norm)

        op_utts.append(_OpUtter(
            idx_in_results=i,
            norm=norm,
            norm2orig=norm2orig,
            global_start=global_start,
            global_end=pos,
        ))

    return "".join(parts), op_utts


def _find_trigger(
    op_stream_norm: str,
    triggers: List[str],
    mode: str,
) -> Optional[Tuple[int, int]]:
    """Find trigger phrase in normalized stream.

    mode='start': pick match with earliest END (cut IVR start as early as possible).
    mode='end':   pick match with earliest START.
    """
    best: Optional[Tuple[int, int]] = None

    for trig in triggers:
        tnorm, _ = _normalize_with_map(trig)
        if not tnorm:
            continue
        s = op_stream_norm.find(tnorm)
        if s == -1:
            continue
        e = s + len(tnorm)

        if best is None:
            best = (s, e)
        elif mode == "start" and e < best[1]:
            best = (s, e)
        elif mode == "end" and s < best[0]:
            best = (s, e)

    return best


def _locate_utterance(op_utts: List[_OpUtter], global_pos: int) -> Optional[int]:
    """Return index of utterance that contains global_pos."""
    if not op_utts:
        return None
    starts = [u.global_start for u in op_utts]
    j = bisect.bisect_right(starts, global_pos) - 1
    if j < 0:
        return None
    u = op_utts[j]
    if u.global_start <= global_pos < u.global_end:
        return j
    return None


def _split_segment_text_by_norm_pos(
    original_text: str,
    norm2orig: List[int],
    split_pos: int,
) -> Tuple[str, str]:
    """Split original_text at normalized position split_pos → (left, right)."""
    if not original_text:
        return "", ""

    split_pos = max(0, min(split_pos, len(norm2orig)))

    if split_pos <= 0:
        return "", original_text
    if split_pos >= len(norm2orig):
        return original_text, ""

    orig_cut = norm2orig[split_pos - 1] + 1
    return original_text[:orig_cut], original_text[orig_cut:]


def relabel_autoresponder_to_ivr(
    results: List[Dict[str, Any]],
    operator_channel: int,
    start_triggers: List[str],
    end_triggers: List[str],
    ivr_channel: int = 3,
) -> List[Dict[str, Any]]:
    """Relabel IVR/autoresponder segments to ivr_channel instead of deleting them.

    START zone: everything before the first start_trigger in operator stream → ivr_channel.
    END zone:   everything after the first end_trigger in operator stream → ivr_channel.
    If a trigger falls mid-utterance, that utterance is split at the trigger boundary.
    """
    out = [dict(r) for r in results]

    # --- START RELABEL ---
    if start_triggers:
        op_stream, op_utts = _build_operator_stream(out, operator_channel)
        m = _find_trigger(op_stream, start_triggers, mode="start")

        if m and op_utts:
            _s, match_end = m
            uj = _locate_utterance(op_utts, max(match_end - 1, 0))

            if uj is not None:
                u = op_utts[uj]
                r_idx = u.idx_in_results
                cut_time = float(out[r_idx]["boundaries"][1])

                local_end = min(max(match_end - u.global_start, 0), len(u.norm))
                orig_txt = out[r_idx].get("transcription") or ""

                if local_end > 0:
                    left_txt, right_txt = _split_segment_text_by_norm_pos(orig_txt, u.norm2orig, local_end)
                else:
                    left_txt, right_txt = "", orig_txt

                left_txt = left_txt.rstrip()
                right_txt = _clean_leading_junk(right_txt)

                new_out: List[Dict[str, Any]] = []
                for k, rr in enumerate(out):
                    rr2 = dict(rr)
                    _b0, b1 = rr2.get("boundaries", (0.0, 0.0))

                    if k == r_idx:
                        if left_txt.strip():
                            ivr_seg = dict(rr2)
                            ivr_seg["channel"] = ivr_channel
                            ivr_seg["transcription"] = left_txt
                            new_out.append(ivr_seg)
                        if right_txt.strip():
                            op_seg = dict(rr2)
                            op_seg["channel"] = operator_channel
                            op_seg["transcription"] = right_txt
                            new_out.append(op_seg)
                        continue

                    if float(b1) <= cut_time:
                        rr2["channel"] = ivr_channel
                    new_out.append(rr2)

                out = new_out

    # --- END RELABEL ---
    if end_triggers:
        op_stream, op_utts = _build_operator_stream(out, operator_channel)
        m = _find_trigger(op_stream, end_triggers, mode="end")

        if m and op_utts:
            match_start, _match_end = m
            uj = _locate_utterance(op_utts, match_start)

            if uj is not None:
                u = op_utts[uj]
                r_idx = u.idx_in_results
                cut_from_time = float(out[r_idx]["boundaries"][0])

                local_start = min(max(match_start - u.global_start, 0), len(u.norm))
                orig_txt = out[r_idx].get("transcription") or ""

                if local_start > 0:
                    left_txt, right_txt = _split_segment_text_by_norm_pos(orig_txt, u.norm2orig, local_start)
                else:
                    left_txt, right_txt = "", orig_txt

                left_txt = left_txt.rstrip()
                right_txt = _clean_leading_junk(right_txt)

                new_out = []
                for k, rr in enumerate(out):
                    rr2 = dict(rr)
                    ch = rr2.get("channel")
                    b0, _b1 = rr2.get("boundaries", (0.0, 0.0))

                    if k == r_idx:
                        if left_txt.strip():
                            op_seg = dict(rr2)
                            op_seg["channel"] = operator_channel
                            op_seg["transcription"] = left_txt
                            new_out.append(op_seg)
                        if right_txt.strip():
                            ivr_seg = dict(rr2)
                            ivr_seg["channel"] = ivr_channel
                            ivr_seg["transcription"] = right_txt
                            new_out.append(ivr_seg)
                        continue

                    if ch == operator_channel and float(b0) >= cut_from_time:
                        rr2["channel"] = ivr_channel
                    new_out.append(rr2)

                out = new_out

    out = [r for r in out if (r.get("transcription") or "").strip()]
    out = sorted(out, key=lambda r: float(r["boundaries"][0]))
    return out


def insert_call_on_hold(
    turns: List[Dict[str, Any]],
    CALL_THRESHOLD: float,
    hold_channel: int = 2,
    hold_tag: str = "[CALL_ON_HOLD]",
) -> List[Dict[str, Any]]:
    """Insert hold markers when silence gap between consecutive turns exceeds CALL_THRESHOLD seconds."""
    if not turns:
        return []

    items = sorted((dict(r) for r in turns), key=lambda r: float(r["boundaries"][0]))
    out: List[Dict[str, Any]] = [items[0]]

    for curr in items[1:]:
        prev = out[-1]
        prev_end = float(prev["boundaries"][1])
        curr_start = float(curr["boundaries"][0])

        if curr_start - prev_end > CALL_THRESHOLD:
            out.append({
                "channel": hold_channel,
                "transcription": hold_tag,
                "boundaries": (prev_end, curr_start),
            })

        out.append(curr)

    return out


def apply_postprocessing(
    turns: List[Dict[str, Any]],
    *,
    operator_channel: int,
    start_triggers: List[str],
    end_triggers: List[str],
    ivr_channel: int = 3,
    hold_channel: int = 2,
    hold_threshold: float = 15.0,
    mask_pii: bool = False,
    spacy_model=None,
) -> List[Dict[str, Any]]:
    """Apply the full postprocessing chain used by service/API code."""
    processed = relabel_autoresponder_to_ivr(
        results=turns,
        operator_channel=operator_channel,
        start_triggers=start_triggers,
        end_triggers=end_triggers,
        ivr_channel=ivr_channel,
    )

    processed = insert_call_on_hold(
        turns=processed,
        CALL_THRESHOLD=hold_threshold,
        hold_channel=hold_channel,
    )

    if mask_pii:
        processed = mask_clean_segments(
            turns=processed,
            spacy_model=spacy_model,
            hold_channel=hold_channel,
            ivr_channel=ivr_channel,
        )

    return processed
