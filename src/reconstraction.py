import torch

from .adapters import CodecAdapter
from .audio import build_analysis_window, fit_audio_shape


def overlap_add_streaming_reconstruct(
    adapter: CodecAdapter,
    wav: torch.Tensor,
    chunk_samples: int,
    hop_samples: int,
    window_type: str,
    scale_mode: str,
) -> torch.Tensor:
    """Chunk wav, reconstruct each chunk independently, and overlap-add."""
    if chunk_samples <= 0 or hop_samples <= 0:
        raise ValueError("chunk_samples and hop_samples must be > 0")

    bsz, channels, total_samples = wav.shape
    if bsz != 1:
        raise ValueError("This script currently expects batch size 1.")

    device = wav.device
    window = build_analysis_window(
        chunk_samples, device=device, window_type=window_type
    )
    window = window.view(1, 1, -1)

    out = torch.zeros((1, channels, total_samples + chunk_samples), device=device)
    weight = torch.zeros((1, 1, total_samples + chunk_samples), device=device)

    start = 0
    while start < total_samples:
        end = start + chunk_samples
        chunk = torch.zeros((1, channels, chunk_samples), device=device)

        valid_end = min(end, total_samples)
        valid_len = valid_end - start
        if valid_len > 0:
            chunk[:, :, :valid_len] = wav[:, :, start:valid_end]

        rec_chunk = adapter.chunk_reconstruct(chunk, scale_mode=scale_mode)
        rec_chunk = fit_audio_shape(
            rec_chunk, target_channels=channels, target_samples=chunk_samples
        )

        out[:, :, start:end] += rec_chunk * window
        weight[:, :, start:end] += window
        start += hop_samples

    out = out[:, :, :total_samples]
    weight = weight[:, :, :total_samples]
    out = out / torch.clamp(weight, min=1e-8)
    return out
