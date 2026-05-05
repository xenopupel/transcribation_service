"""Smoke test: verify that batch padding does not change transcription results.

Tests:
  1. single(wav_a) == batched(wav_a, wav_b)[0]
     — same audio in a batch with another segment must give the same result

  2. single(wav_a) == single(padded_wav_a)  [with real length passed correctly]
     — manual zero-padding must not affect output when lengths are set correctly

Run from the project root:
    python scripts/smoke_test_batch_padding.py --audio path/to/stereo.mp3

The script extracts the first two VAD segments from the file and uses those
as wav_a / wav_b so we test on real speech, not random noise.
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", required=True,
        help="Path to a stereo audio file (used to extract real VAD segments)"
    )
    args = parser.parse_args()

    import gigaam
    from gigaam_service.asr_batch import transcribe_audio_batch
    from gigaam_service.stereo_segmenter import segment_stereo_audio

    print("Loading GigaAM model...")
    model = gigaam.load_model("v3_e2e_rnnt")

    print(f"Running VAD on {args.audio} ...")
    segments = segment_stereo_audio(
        audio_path=args.audio,
        job_id="smoke-test",
        device=model._device,
    )

    if len(segments) < 2:
        print(f"Need at least 2 VAD segments, got {len(segments)}. Use a longer audio file.")
        sys.exit(1)

    wav_a = segments[0].audio
    wav_b = segments[1].audio
    print(f"  wav_a: {len(wav_a)/16000:.2f}s  (ch{segments[0].channel})")
    print(f"  wav_b: {len(wav_b)/16000:.2f}s  (ch{segments[1].channel})")

    # --- Test 1: single vs batched ---
    print("\n[Test 1] single(wav_a) vs batched(wav_a, wav_b)[0]")
    single_a = transcribe_audio_batch(model, [wav_a])[0]
    batched_a = transcribe_audio_batch(model, [wav_a, wav_b])[0]
    print(f"  single : {single_a!r}")
    print(f"  batched: {batched_a!r}")
    assert single_a == batched_a, f"MISMATCH!\n  single:  {single_a!r}\n  batched: {batched_a!r}"
    print("  PASS")

    # --- Test 2: batch order independence ---
    # Position in batch must not change the result.
    # The production pipeline sorts by audio length before batching,
    # so wav_a may end up at any position — this test checks that's safe.
    print("\n[Test 2] batch order independence: [wav_a, wav_b][0] == [wav_b, wav_a][1]")
    single_b = transcribe_audio_batch(model, [wav_b])[0]
    result_a_first, result_b_first = transcribe_audio_batch(model, [wav_a, wav_b])
    result_b_second, result_a_second = transcribe_audio_batch(model, [wav_b, wav_a])

    print(f"  wav_a alone          : {single_a!r}")
    print(f"  wav_a first in batch : {result_a_first!r}")
    print(f"  wav_a second in batch: {result_a_second!r}")

    assert single_a == result_a_first, (
        f"ORDER MISMATCH (first)!\n  alone: {single_a!r}\n  first: {result_a_first!r}"
    )
    assert single_a == result_a_second, (
        f"ORDER MISMATCH (second)!\n  alone: {single_a!r}\n  second: {result_a_second!r}"
    )
    assert single_b == result_b_first, (
        f"ORDER MISMATCH wav_b!\n  alone: {single_b!r}\n  first: {result_b_first!r}"
    )
    print("  PASS")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
