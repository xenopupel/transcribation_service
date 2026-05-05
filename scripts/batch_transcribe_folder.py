"""Transcribe all stereo audio files in a folder.

Usage (CPU):
    python scripts/batch_transcribe_folder.py --input-dir path/to/audio_files --output-dir path/to/out

Usage (GPU):
    python scripts/batch_transcribe_folder.py --input-dir path/to/audio_files --output-dir path/to/out --device cuda

Supported formats: .mp3 .wav .ogg .flac .m4a
Output: one .txt per audio file in --output-dir, plus a timing summary at the end.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Folder with stereo audio files")
    parser.add_argument("--output-dir", required=True, help="Folder to save .txt transcripts")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pause-threshold", type=float, default=2.0)
    parser.add_argument("--strict-limit-duration", type=float, default=30.0)
    parser.add_argument("--hold-threshold", type=float, default=15.0)
    parser.add_argument("--operator-channel", type=int, default=1)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--no-ivr", action="store_true")
    parser.add_argument("--device", default=None,
                        help="'cpu', 'cuda', 'cuda:0', ... Default: cuda if available else cpu")
    args = parser.parse_args()

    import gigaam
    from gigaam_service.formatter import save_transcript_txt
    from gigaam_service.pipeline import transcribe_stereo_file

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(audio_files)} file(s) in {input_dir}")
    print(f"Loading GigaAM model (device={args.device or 'auto'})...")
    model = gigaam.load_model("v3_e2e_rnnt", device=args.device)
    print(f"Model on: {model._device}\n")

    timings: list[tuple[str, float, str]] = []  # (name, seconds, status)

    for idx, audio_path in enumerate(audio_files, 1):
        out_path = output_dir / (audio_path.stem + ".txt")
        print(f"[{idx}/{len(audio_files)}] {audio_path.name} -> {out_path.name}")
        t0 = time.perf_counter()
        status = "OK"
        try:
            result = transcribe_stereo_file(
                audio_path=str(audio_path),
                model=model,
                batch_size=args.batch_size,
                pause_threshold=args.pause_threshold,
                strict_limit_duration=args.strict_limit_duration,
                operator_channel=args.operator_channel,
                apply_postprocess=not args.no_postprocess,
                apply_masking=False,
                hold_threshold=args.hold_threshold,
            )
            save_transcript_txt(
                turns=result["processed_segments"],
                path=str(out_path),
                operator_channel=result["operator_channel"],
                hold_channel=result["hold_channel"],
                ivr_channel=result["ivr_channel"],
                include_ivr=not args.no_ivr,
            )
        except Exception as e:
            status = f"ERROR: {e}"
            print(f"  FAILED: {e}")

        elapsed = time.perf_counter() - t0
        timings.append((audio_path.name, elapsed, status))
        print(f"  {elapsed:.1f}s  [{status}]")

    print("\n--- Summary ---")
    total = sum(t for _, t, _ in timings)
    for name, t, status in timings:
        print(f"  {t:6.1f}s  {status:6}  {name}")
    print(f"  {'':6}  Total: {total:.1f}s for {len(audio_files)} file(s)")


if __name__ == "__main__":
    main()
