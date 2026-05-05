"""CLI: transcribe a single stereo audio file.

Usage:
    python -m gigaam_service.process_file \\
        --audio path/to/audio.wav \\
        --output-json out/result.json \\
        --output-txt out/result.txt \\
        --batch-size 8 \\
        --pause-threshold 2.0 \\
        --strict-limit-duration 30.0 \\
        --hold-threshold 15.0 \\
        --mask-pii
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict

from gigaam_service.formatter import save_transcript_txt
from gigaam_service.models import load_runtime_models
from gigaam_service.pipeline import transcribe_stereo_file

# pyannote tries to load torchcodec native DLLs on import and warns when they're missing.
# The fallback (torchaudio) is used automatically, so this warning is harmless noise.
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tuples → lists so the result is JSON-serializable."""
    out = dict(result)
    for key in ("raw_segments", "processed_segments"):
        segments = []
        for seg in out.get(key, []):
            s = dict(seg)
            if isinstance(s.get("boundaries"), tuple):
                s["boundaries"] = list(s["boundaries"])
            segments.append(s)
        out[key] = segments
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a stereo audio file with GigaAM."
    )
    parser.add_argument("--audio", required=True, help="Path to stereo audio file")
    parser.add_argument("--output-json", help="Path to save full JSON result")
    parser.add_argument("--output-txt", help="Path to save plain-text transcript")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pause-threshold", type=float, default=2.0,
                        help="VAD pause threshold for segment grouping (seconds)")
    parser.add_argument("--strict-limit-duration", type=float, default=30.0,
                        help="Max segment duration fed to ASR (seconds)")
    parser.add_argument("--hold-threshold", type=float, default=15.0,
                        help="Silence gap that triggers [CALL_ON_HOLD] (seconds)")
    parser.add_argument("--operator-channel", type=int, default=1,
                        help="Physical channel index of the operator (default: 1)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Skip IVR relabeling and HOLD insertion")
    parser.add_argument("--no-ivr", action="store_true",
                        help="Exclude IVR segments from the output TXT (JSON always contains them)")
    parser.add_argument("--mask-pii", action="store_true",
                        help="Mask phone numbers and named entities")
    parser.add_argument("--spacy-model", default="ru_core_news_lg",
                        help="spaCy model name for NER masking (default: ru_core_news_lg)")
    parser.add_argument("--device", default=None,
                        help="Device for model inference: 'cpu', 'cuda', 'cuda:0', etc. "
                             "Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    print("Loading runtime models...")
    models = load_runtime_models(
        model_name="v3_e2e_rnnt",
        device=args.device,
        load_spacy=args.mask_pii,
        spacy_model_name=args.spacy_model,
    )

    print(f"Processing: {args.audio}")
    result = transcribe_stereo_file(
        audio_path=args.audio,
        model=models.asr_model,
        batch_size=args.batch_size,
        pause_threshold=args.pause_threshold,
        strict_limit_duration=args.strict_limit_duration,
        operator_channel=args.operator_channel,
        apply_postprocess=not args.no_postprocess,
        apply_masking=args.mask_pii,
        spacy_model=models.spacy_model if args.mask_pii else None,
        hold_threshold=args.hold_threshold,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_serialize_result(result), f, ensure_ascii=False, indent=2)
        print(f"JSON saved: {out_path}")

    if args.output_txt:
        out_path = Path(args.output_txt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_transcript_txt(
            turns=result["processed_segments"],
            path=str(out_path),
            operator_channel=result["operator_channel"],
            hold_channel=result["hold_channel"],
            ivr_channel=result["ivr_channel"],
            include_ivr=not args.no_ivr,
        )
        print(f"TXT saved: {out_path}")

    if not args.output_json and not args.output_txt:
        # Fallback: print to stdout
        for seg in result["processed_segments"]:
            ch = seg["channel"]
            text = seg["transcription"]
            start, end = seg["boundaries"]
            print(f"[{start:.2f}s - {end:.2f}s] Ch{ch}: {text}")


if __name__ == "__main__":
    main()
