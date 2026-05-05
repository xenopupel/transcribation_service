"""Benchmark the current stereo transcription pipeline over a folder.

Runs:
  1. CPU + current custom batching (optional)
  2. GPU + current custom batching (optional)

Reports:
  - Per-file and total wall-clock time for each run
  - Character-level diff between CPU/GPU runs when both are enabled

Usage:
    python scripts/benchmark_batching.py --input-dir path/to/audio_files
"""

import argparse
import difflib
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message="triton not found")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


@dataclass
class RunResult:
    label: str
    timings: Dict[str, float] = field(default_factory=dict)   # filename → seconds
    transcripts: Dict[str, str] = field(default_factory=dict) # filename → full txt


def _join_transcript(processed_segments, operator_channel, hold_channel, ivr_channel) -> str:
    """Build a plain string from processed segments (mirrors formatter logic)."""
    lines = []
    for seg in processed_segments:
        ch = seg["channel"]
        if ch == ivr_channel:
            continue
        text = seg["transcription"].strip()
        if not text:
            continue
        lines.append(text)
    return "\n".join(lines)


def _run_folder(
    audio_files: List[Path],
    model,
    batch_size: int,
    pause_threshold: float,
    strict_limit_duration: float,
    label: str,
) -> RunResult:
    from gigaam_service.pipeline import transcribe_stereo_file
    from gigaam_service import HOLD_CHANNEL, IVR_CHANNEL, OPERATOR_CHANNEL

    result = RunResult(label=label)
    print(f"\n{'='*60}")
    print(f"  Run: {label}")
    print(f"{'='*60}")

    for audio_path in audio_files:
        print(f"  {audio_path.name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            out = transcribe_stereo_file(
                audio_path=str(audio_path),
                model=model,
                batch_size=batch_size,
                pause_threshold=pause_threshold,
                strict_limit_duration=strict_limit_duration,
                operator_channel=OPERATOR_CHANNEL,
                apply_postprocess=True,
                apply_masking=False,
            )
            elapsed = time.perf_counter() - t0
            txt = _join_transcript(
                out["processed_segments"],
                out["operator_channel"],
                out["hold_channel"],
                out["ivr_channel"],
            )
            result.timings[audio_path.name] = elapsed
            result.transcripts[audio_path.name] = txt
            print(f"{elapsed:.1f}s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            result.timings[audio_path.name] = elapsed
            result.transcripts[audio_path.name] = f"ERROR: {e}"
            print(f"ERROR: {e}")
            traceback.print_exc()

    total = sum(result.timings.values())
    print(f"  --- total: {total:.1f}s ---")
    return result


def _print_timing_table(runs: List[RunResult], audio_files: List[Path]) -> None:
    filenames = [f.name for f in audio_files]
    col = max(len(n) for n in filenames) + 2
    header = f"{'File':<{col}}" + "".join(f"{r.label:>16}" for r in runs)
    print(header)
    print("-" * len(header))
    for name in filenames:
        row = f"{name:<{col}}"
        for r in runs:
            t = r.timings.get(name, -1)
            row += f"{t:>14.1f}s"
        print(row)
    print("-" * len(header))
    total_row = f"{'TOTAL':<{col}}"
    for r in runs:
        total_row += f"{sum(r.timings.values()):>14.1f}s"
    print(total_row)


def _char_diff_ratio(a: str, b: str) -> float:
    """0.0 = identical, 1.0 = completely different."""
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def _print_diff_table(runs: List[RunResult], audio_files: List[Path]) -> None:
    """Print pairwise character-diff ratios between all run pairs."""
    if len(runs) < 2:
        print("  Need at least two runs to compare transcripts.")
        return

    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairs.append((runs[i], runs[j]))

    filenames = [f.name for f in audio_files]
    col = max(len(n) for n in filenames) + 2
    pair_labels = [f"{a.label} vs {b.label}" for a, b in pairs]
    header = f"{'File':<{col}}" + "".join(f"{lbl:>26}" for lbl in pair_labels)
    print(header)
    print("-" * len(header))

    any_mismatch = False
    for name in filenames:
        row = f"{name:<{col}}"
        file_mismatch = False
        for a, b in pairs:
            ta = a.transcripts.get(name, "")
            tb = b.transcripts.get(name, "")
            ratio = _char_diff_ratio(ta, tb)
            marker = " *" if ratio > 0.0 else "  "
            row += f"{ratio:>22.4f}{marker}"
            if ratio > 0.0:
                file_mismatch = True
                any_mismatch = True
        print(row)
        if file_mismatch:
            _print_first_diff(name, runs)

    print("-" * len(header))
    if not any_mismatch:
        print("  All runs produced identical transcripts.")
    else:
        print("  * = outputs differ between these two runs")


def _print_first_diff(filename: str, runs: List[RunResult]) -> None:
    """Show unified diff for the first pair that differs."""
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a_txt = runs[i].transcripts.get(filename, "")
            b_txt = runs[j].transcripts.get(filename, "")
            if a_txt == b_txt:
                continue
            a_lines = a_txt.splitlines(keepends=True)
            b_lines = b_txt.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                a_lines, b_lines,
                fromfile=runs[i].label, tofile=runs[j].label,
                lineterm="",
            ))
            if diff:
                print(f"    [diff {runs[i].label} vs {runs[j].label} — {filename}]")
                for line in diff[:40]:  # cap at 40 lines
                    print(f"      {line.rstrip()}")
                if len(diff) > 40:
                    print(f"      ... ({len(diff) - 40} more lines)")
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pause-threshold", type=float, default=2.0)
    parser.add_argument("--strict-limit-duration", type=float, default=30.0)
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU runs (faster iteration)")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU runs")
    args = parser.parse_args()

    import gigaam

    input_dir = Path(args.input_dir)
    audio_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(audio_files)} file(s):")
    for f in audio_files:
        print(f"  {f.name}")

    runs: List[RunResult] = []

    # ── CPU runs ──────────────────────────────────────────────────
    if not args.skip_cpu:
        print("\nLoading model on CPU...")
        model_cpu = gigaam.load_model("v3_e2e_rnnt", device="cpu")
        print(f"  device: {model_cpu._device}")

        runs.append(_run_folder(
            audio_files, model_cpu,
            batch_size=args.batch_size,
            pause_threshold=args.pause_threshold,
            strict_limit_duration=args.strict_limit_duration,
            label="CPU",
        ))
        del model_cpu

    # ── GPU runs ──────────────────────────────────────────────────
    if not args.skip_gpu:
        import torch
        if not torch.cuda.is_available():
            print("\nCUDA not available — skipping GPU runs.")
        else:
            print("\nLoading model on GPU...")
            model_gpu = gigaam.load_model("v3_e2e_rnnt", device="cuda")
            print(f"  device: {model_gpu._device}  ({torch.cuda.get_device_name(0)})")

            runs.append(_run_folder(
                audio_files, model_gpu,
                batch_size=args.batch_size,
                pause_threshold=args.pause_threshold,
                strict_limit_duration=args.strict_limit_duration,
                label="GPU",
            ))
            del model_gpu

    if not runs:
        print("No runs to compare.")
        sys.exit(0)

    # ── Results ───────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  TIMING  (seconds per file)")
    print("="*60)
    _print_timing_table(runs, audio_files)

    print("\n\n" + "="*60)
    print("  CHARACTER DIFF  (0.0 = identical)")
    print("="*60)
    _print_diff_table(runs, audio_files)


if __name__ == "__main__":
    main()
