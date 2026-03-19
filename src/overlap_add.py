"""
overlap_add.py
==============
Overlap-add reconstruction from pre-encoded EnCodec frames.

Instead of encoding + decoding inside the loop, this module receives
already-encoded frames (codes + scales) and reconstructs each one through
model.decode, then recombines the decoded chunks with overlap-add synthesis.

                  chunk 0       chunk 1       chunk 2
    |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|             |             |
    |      |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|             |
    |      |      |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
    0    hop     2*hop    ...            total_samples
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from encodec import EncodecModel


# Type alias: one encoded frame = (codes [B, K, T], scale [B, 1] or None)
EncodedFrame = Tuple[torch.Tensor, Optional[torch.Tensor]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_chunks(
    model: EncodecModel,
    wav: torch.Tensor,
    chunk_samples: int,
    hop_samples: int,
) -> List[EncodedFrame]:
    """
    Slide a window over *wav* [1, C, T], encode each chunk independently,
    and return a list of EncodedFrames suitable for overlap_add_decode.

    The last chunk is zero-padded to chunk_samples if shorter than a full chunk.

    Parameters
    ----------
    model         : EncodecModel
    wav           : [1, C, T]  input waveform (batch size 1)
    chunk_samples : int  window length in samples
    hop_samples   : int  step between consecutive windows

    Returns
    -------
    List of (codes [B, K, T_frames], scale [B, 1] | None)
    """
    if chunk_samples <= 0 or hop_samples <= 0:
        raise ValueError("chunk_samples and hop_samples must be > 0")

    total_samples = wav.shape[-1]
    frames: List[EncodedFrame] = []

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = wav[:, :, start:end]
        if chunk.shape[-1] < chunk_samples:
            chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
        chunk_frames = model.encode(chunk)
        frames.append(chunk_frames[0])
        start += hop_samples

    return frames

def overlap_add_decode(
    model: EncodecModel,
    frames: List[EncodedFrame],
    chunk_samples: int,
    hop_samples: int,
    total_samples: int,
    channels: int = 2,
    window_type: str = "hann",
) -> torch.Tensor:
    """
    Decode a list of EncodedFrames and recombine with overlap-add synthesis.

    Each frame is decoded independently via ``model.decode``, windowed, and
    accumulated into the output buffer.  The weight accumulator tracks
    coverage so the final division normalises correctly at all positions,
    including boundaries.

    Parameters
    ----------
    model         : EncodecModel
        The EnCodec model used for decoding  (e.g. encodec_model_48khz).
    frames        : list of (codes, scale)
        Pre-encoded frames in the same format returned by ``model.encode``.
        codes : [B, K, T_frames]
        scale : [B, 1] or None
    chunk_samples : int
        Length of each decoded chunk in samples.  Must match the chunk
        size used during encoding.
    hop_samples   : int
        Step size between consecutive chunk positions in samples.
        hop_samples < chunk_samples produces overlap.
    total_samples : int
        Length of the original waveform in samples.  Output is trimmed to
        this length.
    channels      : int
        Number of output audio channels  (1 = mono, 2 = stereo).
    window_type   : str
        Synthesis window: ``"hann"`` (recommended) or ``"rect"`` (no window).

    Returns
    -------
    torch.Tensor  [1, channels, total_samples]
        Reconstructed waveform.

    Raises
    ------
    ValueError
        If chunk_samples or hop_samples are not positive, if
        hop_samples > chunk_samples (which would leave gaps), or if
        frames is empty.

    Notes
    -----
    Hann window + 50% overlap (hop = chunk // 2) gives the smoothest
    reconstruction.  For EnCodec's native 1% overlap use:
        chunk_samples = 48000
        hop_samples   = 47520   (48000 - 480)
    """
    _validate(chunk_samples, hop_samples, frames)

    device = frames[0][0].device
    window = _make_window(chunk_samples, device, window_type)   # [1, 1, chunk_samples]

    out, weight = _make_buffers(channels, total_samples, chunk_samples, device)

    for i, frame in enumerate(frames):
        start = i * hop_samples
        end   = start + chunk_samples

        # Decode a single frame: pass it as a one-element list, the format
        # model.decode expects → returns [B, C, T_decoded]
        rec = model.decode([frame])
        rec = _fit(rec, channels, chunk_samples, device)

        out[:, :, start:end]    += rec * window
        weight[:, :, start:end] += window

    return _normalise(out, weight, total_samples)


# ---------------------------------------------------------------------------
# Window factory
# ---------------------------------------------------------------------------

def _make_window(
    chunk_samples: int,
    device: torch.device,
    window_type: str,
) -> torch.Tensor:
    """
    Build a [1, 1, chunk_samples] synthesis window.

    ``"hann"``  — standard Hann window, recommended for smooth OLA.
                  Requires 50% overlap (hop = chunk // 2) for perfect
                  reconstruction from the window alone.
    ``"rect"``  — rectangular (all ones), no windowing effect.
    """
    if window_type == "hann":
        w = torch.hann_window(chunk_samples, device=device)
    elif window_type == "rect":
        w = torch.ones(chunk_samples, device=device)
    else:
        raise ValueError(
            f"Unknown window_type '{window_type}'. Choose 'hann' or 'rect'."
        )
    return w.view(1, 1, -1)


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def _make_buffers(
    channels: int,
    total_samples: int,
    chunk_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Allocate the output accumulator [1, C, T+chunk] and weight
    accumulator [1, 1, T+chunk].

    The extra chunk_samples of padding ensures the last frame never
    writes out of bounds regardless of alignment.
    """
    padded = total_samples + chunk_samples
    out    = torch.zeros((1, channels, padded), device=device)
    weight = torch.zeros((1, 1,        padded), device=device)
    return out, weight


def _fit(
    rec: torch.Tensor,
    channels: int,
    chunk_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Ensure the decoded chunk is exactly [1, channels, chunk_samples].

    model.decode may return a slightly different length due to transposed
    convolution padding — trim or zero-pad as needed.
    """
    # Fix channel count
    if rec.shape[1] < channels:
        rec = rec.expand(-1, channels, -1).contiguous()
    elif rec.shape[1] > channels:
        rec = rec[:, :channels, :]

    # Fix time dimension
    T = rec.shape[-1]
    if T > chunk_samples:
        rec = rec[:, :, :chunk_samples]
    elif T < chunk_samples:
        rec = F.pad(rec, (0, chunk_samples - T))

    return rec


def _normalise(
    out: torch.Tensor,
    weight: torch.Tensor,
    total_samples: int,
) -> torch.Tensor:
    """Trim to total_samples and normalise by the weight accumulator."""
    out    = out[:, :, :total_samples]
    weight = weight[:, :, :total_samples]
    return out / weight.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(
    chunk_samples: int,
    hop_samples: int,
    frames: List[EncodedFrame],
) -> None:
    if chunk_samples <= 0:
        raise ValueError(f"chunk_samples must be > 0, got {chunk_samples}")
    if hop_samples <= 0:
        raise ValueError(f"hop_samples must be > 0, got {hop_samples}")
    if hop_samples > chunk_samples:
        raise ValueError(
            f"hop_samples ({hop_samples}) > chunk_samples ({chunk_samples}) "
            "would leave gaps in the output."
        )
    if len(frames) == 0:
        raise ValueError("frames list is empty.")
