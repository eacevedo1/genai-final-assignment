from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

# ── src/ functions ────────────────────────────────────────────────────────────
from src.device import get_device                              # auto-selects best accelerator
from src.eval import evaluate_pair, print_results              # waveform quality metrics
from src.overlap_add import overlap_add_decode, encode_chunks  # OLA from encoded frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming codec demo using new src/ package functions."
    )
    parser.add_argument("input_audio", type=Path, help="Input audio file path.")
    parser.add_argument(
        "--codec",
        type=str,
        default="encodec48",
        choices=["encodec48", "encodec24"],
        help="EnCodec model variant.",
    )
    parser.add_argument("--bandwidth", type=float, default=12.0, help="Encodec bandwidth in kbps.")
    parser.add_argument("--chunk-seconds", type=float, default=1.0, help="Streaming chunk size in seconds.")
    parser.add_argument("--hop-seconds", type=float, default=0.5, help="Streaming hop size in seconds.")
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        choices=["hann", "rect"],
        help="Synthesis window for overlap-add reconstruction.",
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

    # Load EnCodec model directly
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

    # ── full reconstruction: encode then decode ───────────────────────────────
    with torch.no_grad():
        frames = model.encode(wav)
        full_reconstruction = model.decode(frames)

    # ── streaming overlap-add reconstruction ─────────────────────────────────
    with torch.no_grad():
        chunk_frames = encode_chunks(model, wav, chunk_samples, hop_samples)
    streaming_reconstruction = overlap_add_decode(
        model,
        chunk_frames,
        chunk_samples=chunk_samples,
        hop_samples=hop_samples,
        total_samples=total_samples,
        channels=model.channels,
        window_type=args.window,
    )

    # ── evaluate with src.eval metrics ───────────────────────────────────────
    results = [
        evaluate_pair(wav, full_reconstruction,      label="Full reconstruct"),
        evaluate_pair(wav, streaming_reconstruction, label="Streaming OLA"),
    ]
    print_results(results, reference_label="Full reconstruct")

    # ── save outputs ──────────────────────────────────────────────────────────
    stem = args.input_audio.stem
    full_output   = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_full.wav"
    stream_output = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_stream.wav"

    def _save(path: Path, audio: torch.Tensor) -> None:
        wav_2d = audio.squeeze(0).detach().cpu().float()
        torchaudio.save(str(path), wav_2d, model.sample_rate)

    _save(full_output,   full_reconstruction)
    _save(stream_output, streaming_reconstruction)

    print(f"Saved full reconstruction  → {full_output}")
    print(f"Saved streaming OLA output → {stream_output}")


if __name__ == "__main__":
    main()
