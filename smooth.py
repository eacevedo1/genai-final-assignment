"""
smooth.py
=========
Streaming overlap-add reconstruction with exponential smoothing of the
per-chunk EnCodec scale factor.

Background
----------
EnCodec 48 kHz normalises each chunk to unit RMS before encoding and stores
the raw scale value alongside the codes so the decoder can undo the
normalisation.  When consecutive chunks have very different loudness levels the
hard per-chunk scale creates a sudden amplitude jump at chunk boundaries that
is audible even with overlap-add synthesis.

This script encodes the input audio as overlapping chunks (via
``src.overlap_add.encode_chunks``), applies EMA smoothing to the per-chunk
scale factors, and reconstructs with overlap-add (``src.overlap_add.overlap_add_decode``).
It compares four reconstructions and reports SI-SDR / SNR / L1 / MSE metrics:

  1. Full reconstruct   — standard encode + decode, no chunks
  2. Streaming OLA      — chunked encode + overlap-add decode, no smoothing
  3. Linear EMA         — chunked, scale smoothed in the amplitude domain
  4. Log EMA            — chunked, scale smoothed in the log domain (exp back)

Smoothing modes
---------------
``linear``  (default)
    EMA directly on the scale value:

        s_hat[t] = alpha * s[t] + (1 - alpha) * s_hat[t-1]

``log``
    EMA in the log domain, then exponentiate back:

        g[t]   = alpha * log(s[t]) + (1 - alpha) * g[t-1]
        out[t] = exp(g[t])

    Numerically better when consecutive chunk scales span several orders of
    magnitude (e.g. a very quiet chunk followed by a very loud one).

Alpha guide
-----------
  alpha = 1.0  →  original hard per-chunk behaviour (no smoothing)
  alpha = 0.5  →  balanced: adapts in ~2 chunks
  alpha = 0.1  →  heavy smoothing: adapts slowly, very stable across chunks
  alpha = 0.3  →  recommended starting point for music

Usage
-----
    python smooth.py <input_audio> [options]

    python smooth.py data/audio/000/000002.wav \\
        --codec encodec48 --bandwidth 6.0 \\
        --chunk-seconds 0.5 --hop-seconds 0.25 --window hann \\
        --alpha 0.3

    # Compare multiple alpha values for both smoothing modes:
    python smooth.py data/audio/000/000002.wav \\
        --codec encodec48 --alpha 0.3 \\
        --compare-alphas 1.0 0.5 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

# ── src/ functions ────────────────────────────────────────────────────────────
from src.device import get_device                              # auto-selects best accelerator
from src.eval import evaluate_pair, print_results              # waveform quality metrics
from src.audio import load_audio                               # load & resample audio
from src.overlap_add import encode_chunks, overlap_add_decode  # chunk encode + OLA decode


EncodedFrame = Tuple[torch.Tensor, Optional[torch.Tensor]]  # (codes, scale)


# ---------------------------------------------------------------------------
# SmoothedScaleCodec
# ---------------------------------------------------------------------------

class SmoothedScaleCodec:
    """
    Thin wrapper around an EncodecModel that applies exponential smoothing
    to the per-chunk scale produced during encoding, and uses those smoothed
    scales during decoding.

    Parameters
    ----------
    model : EncodecModel
        A pre-loaded EnCodec model (intended for the 48 kHz stereo variant but
        also works with the 24 kHz model).
    alpha : float
        Smoothing factor in (0, 1].  Higher → faster adaptation (closer to raw
        per-chunk behaviour).  Lower → heavier smoothing.
    """

    def __init__(self, model: EncodecModel, alpha: float = 0.3, mode: str = "linear"):
        assert 0.0 < alpha <= 1.0, "alpha must be in (0, 1]"
        assert mode in ("linear", "log"), "mode must be 'linear' or 'log'"
        self.model = model
        self.alpha = alpha
        self.mode  = mode

    # Reuse src.audio.load_audio as a method (signature: (self, path) → [1, C, T])
    load_audio = load_audio

    # ------------------------------------------------------------------
    # Core encode  (returns smoothed scales instead of raw chunk scales)
    # ------------------------------------------------------------------
    def encode(self, wav: torch.Tensor) -> List[EncodedFrame]:
        """
        Encode `wav` into a list of (codes, smoothed_scale) tuples.

        Parameters
        ----------
        wav : torch.Tensor  shape [B, C, T]
            Audio waveform already resampled and channel-matched to the model.

        Returns
        -------
        List of (codes, smoothed_scale) — same format as EncodecModel.encode()
        but with smoothed scales.
        """
        with torch.no_grad():
            raw_frames: List[EncodedFrame] = self.model.encode(wav)
        return self.smooth_frames(raw_frames)

    # ------------------------------------------------------------------
    # Core decode
    # ------------------------------------------------------------------
    def decode(self, frames: List[EncodedFrame]) -> torch.Tensor:
        """
        Decode a list of (codes, scale) frames back to a waveform.

        Parameters
        ----------
        frames : list of (codes, scale)
            Typically produced by `self.encode()`.

        Returns
        -------
        torch.Tensor  shape [B, C, T]
        """
        with torch.no_grad():
            return self.model.decode(frames)

    # ------------------------------------------------------------------
    # Convenience: encode + decode in one call
    # ------------------------------------------------------------------
    def encode_decode(self, wav: torch.Tensor) -> torch.Tensor:
        """Round-trip encode then decode, returning the reconstructed waveform."""
        return self.decode(self.encode(wav))

    # ------------------------------------------------------------------
    # Public: apply EMA to any list of encoded frames
    # ------------------------------------------------------------------
    def smooth_frames(self, raw_frames: List[EncodedFrame]) -> List[EncodedFrame]:
        """
        Walk through a list of (codes, scale) frames — e.g. produced by
        ``encode_chunks`` — and replace each scale with its EMA value.

        Two modes are supported (set via ``self.mode``):

        ``"linear"``  (default)
            EMA in the amplitude domain:
            ema[t] = alpha * scale[t] + (1 - alpha) * ema[t-1]

        ``"log"``
            EMA in the log domain, then exponentiate back:
            g[t]   = alpha * log(scale[t]) + (1 - alpha) * g[t-1]
            out[t] = exp(g[t])

            This is numerically better when consecutive chunk scales span
            several orders of magnitude.

        The EMA / log-EMA is initialised from the first chunk's scale so the
        very first chunk is never distorted.
        """
        if not raw_frames:
            return raw_frames

        alpha = self.alpha
        log_mode = self.mode == "log"
        smoothed: List[EncodedFrame] = []
        ema: Optional[torch.Tensor] = None  # stores linear EMA or log-EMA accumulator

        for codes, scale in raw_frames:
            if scale is None:
                smoothed.append((codes, scale))
                continue

            if log_mode:
                log_scale = torch.log(scale.abs().clamp(min=1e-8))
                if ema is None:
                    ema = log_scale.clone()
                else:
                    # g[t] = alpha * log(scale[t]) + (1 - alpha) * g[t-1]
                    ema = alpha * log_scale + (1.0 - alpha) * ema
                out_scale = torch.exp(ema)
            else:
                if ema is None:
                    ema = scale.clone()
                else:
                    # s_hat[t] = alpha * s[t] + (1 - alpha) * s_hat[t-1]
                    ema = alpha * scale + (1.0 - alpha) * ema
                out_scale = ema

            smoothed.append((codes, out_scale.clone()))

        return smoothed


# ---------------------------------------------------------------------------
# Alpha comparison (chunk-aware)
# ---------------------------------------------------------------------------

def compare_alpha_values(
    wav: torch.Tensor,
    model: EncodecModel,
    alphas: List[float],
    chunk_samples: int,
    hop_samples: int,
    total_samples: int,
    window_type: str = "hann",
    mode: str = "linear",
) -> None:
    """
    Encode wav as chunks, then decode with each alpha value and print a
    comparison table using src.eval metrics.
    """
    with torch.no_grad():
        chunk_frames = encode_chunks(model, wav, chunk_samples, hop_samples)

    results = []
    for alpha in alphas:
        codec = SmoothedScaleCodec(model, alpha=alpha, mode=mode)
        smoothed = codec.smooth_frames(chunk_frames)
        recon = overlap_add_decode(
            model, smoothed,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            total_samples=total_samples,
            channels=model.channels,
            window_type=window_type,
        )
        results.append(evaluate_pair(wav, recon, label=f"alpha={alpha:.2f} ({mode})"))
    print_results(results, reference_label=f"alpha={alphas[0]:.2f} ({mode})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoothed-scale EnCodec demo: exponential EMA smoothing of per-chunk scale."
    )
    parser.add_argument("input_audio", type=Path, help="Input audio file path.")
    parser.add_argument(
        "--codec",
        type=str,
        default="encodec48",
        choices=["encodec48", "encodec24"],
        help="EnCodec model variant.",
    )
    parser.add_argument("--bandwidth", type=float, default=6.0, help="Encodec bandwidth in kbps.")
    parser.add_argument("--chunk-seconds", type=float, default=1.0, help="Streaming chunk size in seconds.")
    parser.add_argument("--hop-seconds", type=float, default=0.5, help="Streaming hop size in seconds.")
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        choices=["hann", "rect"],
        help="Synthesis window for overlap-add reconstruction.",
    )
    parser.add_argument("--alpha", type=float, default=0.3, help="EMA smoothing factor in (0, 1].")
    parser.add_argument(
        "--smooth-mode",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="Smoothing domain: 'linear' (EMA on scale) or 'log' (EMA on log(scale), then exp).",
    )
    parser.add_argument(
        "--compare-alphas",
        type=float,
        nargs="+",
        default=None,
        metavar="ALPHA",
        help="Extra alpha values to include in a comparison table (e.g. --compare-alphas 1.0 0.5 0.1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # src.device.get_device() auto-selects CUDA → MPS → CPU
    device = torch.device(get_device() if args.device == "auto" else args.device)

    # Load EnCodec model
    if args.codec == "encodec48":
        model = EncodecModel.encodec_model_48khz()
    else:
        model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(args.bandwidth)
    model.to(device).eval()

    # Load and prepare audio
    wav_raw, sr = torchaudio.load(str(args.input_audio))
    wav_raw = convert_audio(wav_raw, sr, model.sample_rate, model.channels)
    wav = wav_raw.unsqueeze(0).to(device)  # [1, C, T]

    chunk_samples = max(1, int(round(args.chunk_seconds * model.sample_rate)))
    hop_samples   = max(1, int(round(args.hop_seconds   * model.sample_rate)))
    if hop_samples > chunk_samples:
        raise ValueError("hop_seconds must be ≤ chunk_seconds")
    total_samples = wav.shape[-1]

    # ── full reconstruction baseline (no chunks) ──────────────────────────────
    with torch.no_grad():
        baseline_frames = model.encode(wav)
        full_recon      = model.decode(baseline_frames)

    # ── streaming OLA: encode chunks, no smoothing ───────────────────────────
    with torch.no_grad():
        chunk_frames = encode_chunks(model, wav, chunk_samples, hop_samples)
    streaming_recon = overlap_add_decode(
        model, chunk_frames,
        chunk_samples=chunk_samples,
        hop_samples=hop_samples,
        total_samples=total_samples,
        channels=model.channels,
        window_type=args.window,
    )

    # ── streaming OLA: linear EMA smoothing ──────────────────────────────────
    linear_codec  = SmoothedScaleCodec(model, alpha=args.alpha, mode="linear")
    linear_frames = linear_codec.smooth_frames(chunk_frames)
    linear_recon  = overlap_add_decode(
        model, linear_frames,
        chunk_samples=chunk_samples,
        hop_samples=hop_samples,
        total_samples=total_samples,
        channels=model.channels,
        window_type=args.window,
    )

    # ── streaming OLA: log-domain EMA smoothing ───────────────────────────────
    log_codec  = SmoothedScaleCodec(model, alpha=args.alpha, mode="log")
    log_frames = log_codec.smooth_frames(chunk_frames)
    log_recon  = overlap_add_decode(
        model, log_frames,
        chunk_samples=chunk_samples,
        hop_samples=hop_samples,
        total_samples=total_samples,
        channels=model.channels,
        window_type=args.window,
    )

    # ── evaluate with src.eval metrics ───────────────────────────────────────
    results = [
        evaluate_pair(wav, full_recon,      label="Full reconstruct"),
        evaluate_pair(wav, streaming_recon, label="Streaming OLA"),
        evaluate_pair(wav, linear_recon,    label=f"Linear EMA  (alpha={args.alpha:.2f})"),
        evaluate_pair(wav, log_recon,       label=f"Log EMA     (alpha={args.alpha:.2f})"),
    ]
    print_results(results, reference_label="Full reconstruct")

    # ── optional multi-alpha comparison ──────────────────────────────────────
    if args.compare_alphas:
        alphas = sorted({1.0, args.alpha, *args.compare_alphas}, reverse=True)
        print("\nAlpha comparison — linear smoothing:")
        compare_alpha_values(
            wav, model, alphas,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            total_samples=total_samples,
            window_type=args.window,
            mode="linear",
        )
        print("\nAlpha comparison — log smoothing:")
        compare_alpha_values(
            wav, model, alphas,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            total_samples=total_samples,
            window_type=args.window,
            mode="log",
        )

    # ── save outputs ──────────────────────────────────────────────────────────
    stem = args.input_audio.stem
    full_output   = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_full.wav"
    stream_output = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_stream.wav"
    linear_output = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_stream_linear_alpha{args.alpha:g}.wav"
    log_output    = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_stream_log_alpha{args.alpha:g}.wav"

    def _save(path: Path, audio: torch.Tensor) -> None:
        wav_2d = audio.squeeze(0).detach().cpu().float()
        torchaudio.save(str(path), wav_2d, model.sample_rate)

    _save(full_output,   full_recon)
    _save(stream_output, streaming_recon)
    _save(linear_output, linear_recon)
    _save(log_output,    log_recon)

    print(f"Saved full reconstruction          → {full_output}")
    print(f"Saved streaming OLA                → {stream_output}")
    print(f"Saved linear EMA smoothed OLA      → {linear_output}")
    print(f"Saved log EMA smoothed OLA         → {log_output}")


if __name__ == "__main__":
    main()