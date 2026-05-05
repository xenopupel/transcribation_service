"""ASR batch inference over a list of 1D audio tensors.

Handles padding and length tracking internally.
Calls model.forward() and model.decoding.decode() — no ONNX, no Triton.
"""

from typing import List

import torch


@torch.inference_mode()
def transcribe_audio_batch(model, audios: List[torch.Tensor]) -> List[str]:
    """Transcribe a list of 1D waveform tensors in a single forward pass.

    Tensors may have different lengths — padding is applied internally.
    Real lengths are captured BEFORE padding so the encoder gets correct masks.

    Args:
        model:  GigaAMASR instance (must have .forward(), .decoding, .head, ._device, ._dtype)
        audios: list of 1D float tensors, any length

    Returns:
        list of transcription strings, same order as input
    """
    if not audios:
        return []

    model.eval()
    device = model._device
    dtype = model._dtype

    # Normalize to 1D and move to correct device/dtype
    prepared: List[torch.Tensor] = []
    for a in audios:
        a = a.to(device=device, dtype=dtype)
        if a.dim() > 1:
            a = a.squeeze()
        prepared.append(a)

    # Real lengths before padding
    lengths = [len(a) for a in prepared]

    # 320 samples is the minimum the preprocessor accepts
    max_len = max(max(lengths), 320)

    batch = torch.zeros(len(prepared), max_len, dtype=dtype, device=device)
    for i, audio in enumerate(prepared):
        batch[i, :len(audio)] = audio

    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    encoded, encoded_len = model.forward(batch, lengths_tensor)
    return model.decoding.decode(model.head, encoded, encoded_len)
