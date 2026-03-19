"""
learned_scale_predictor.py
==========================
Streaming overlap-add reconstruction with a learned causal GRU scale predictor
for EnCodec 48 kHz.

Concept
-------
EnCodec 48 kHz normalises each chunk to unit RMS before encoding and stores
the raw scale alongside the codes so the decoder can undo the normalisation.
This requires the model to "see" the full chunk before encoding, creating a
look-ahead.

Here, a small causal GRU reads the encoder's latent frames for the current
chunk and predicts what the scale should be — without looking at future chunks.
The hidden state is carried across chunks, giving the predictor memory of the
loudness trajectory of the whole signal.  The predicted scale is then used
inside an overlap-add (OLA) streaming pipeline.

Architecture
------------

    EnCodec encoder  →  latent z  [B, 128, T_frames]
                                ↓
                      ScalePredictor (GRU)
                    hidden ──────────────► carried to next chunk
                                ↓
                      predicted log-scale [B, 1]
                                ↓ exp()
                      overlap_add_decode  (src.overlap_add)

Key numbers for 48 kHz
-----------------------
  encoder output (hidden_size) : 128 channels
  frames per 1-second chunk    : 150  (48000 / 320 hop)
  scale per chunk              : scalar [B, 1]  (stored as log for training)

Compares three reconstructions and prints SI-SDR / SNR / L1 / MSE metrics:
  1. Full reconstruct        — standard encode + decode, no chunks
  2. Streaming OLA           — chunked encode + overlap-add decode, raw scale
  3. Predicted-scale OLA     — chunked encode + OLA with GRU-predicted scale

Usage
-----
    # Evaluate on a single file (requires a trained predictor checkpoint):
    python learned_scale_predictor.py eval data/audio/000/000002.wav predictor.pt \\
        --codec encodec48 --chunk-seconds 0.5 --hop-seconds 0.25 --window hann

    # Train on a folder of stereo 48 kHz .wav files:
    python learned_scale_predictor.py train ./music_wavs/ predictor.pt

    # Architecture smoke test (no audio or checkpoint needed):
    python learned_scale_predictor.py demo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset

from encodec import EncodecModel
from encodec.utils import convert_audio

# ── src/ functions ────────────────────────────────────────────────────────────
from src.device import get_device                              # auto-selects best accelerator
from src.eval import evaluate_pair, print_results              # waveform quality metrics
from src.audio import load_audio                               # load & resample audio
from src.overlap_add import encode_chunks, overlap_add_decode  # chunk encode + OLA decode


# ---------------------------------------------------------------------------
# Constants for 48 kHz model
# ---------------------------------------------------------------------------
SAMPLE_RATE     = 48_000
CHANNELS        = 2
HIDDEN_SIZE     = 128            # encoder output channels for 48 kHz
CHUNK_SAMPLES   = 48_000         # 1-second chunks
OVERLAP_RATIO   = 0.01
OVERLAP_SAMPLES = int(CHUNK_SAMPLES * OVERLAP_RATIO)   # 480
STRIDE_SAMPLES  = CHUNK_SAMPLES - OVERLAP_SAMPLES      # 47 520


# ---------------------------------------------------------------------------
# 1.  Scale Predictor
# ---------------------------------------------------------------------------

class ScalePredictor(nn.Module):
    """
    Causal GRU that maps one chunk of encoder latent frames to a predicted
    log-scale scalar.

    Parameters
    ----------
    input_dim  : encoder output channels  (128 for 48 kHz)
    hidden_dim : GRU hidden size  (64 is enough for a scalar regression task)
    num_layers : stacked GRU layers
    """

    def __init__(
        self,
        input_dim: int = HIDDEN_SIZE,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,          # input: [B, T_frames, input_dim]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        latent_chunk: torch.Tensor,              # [B, input_dim, T_frames]
        hidden: Optional[torch.Tensor] = None,   # GRU hidden state
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        latent_chunk : [B, input_dim, T_frames]
        hidden       : previous GRU hidden state, or None for first chunk

        Returns
        -------
        log_scale : [B, 1]   predicted log-scale for this chunk
        hidden    : updated hidden state — pass to the next chunk
        """
        x            = latent_chunk.permute(0, 2, 1)   # [B, T_frames, input_dim]
        out, hidden  = self.gru(x, hidden)              # [B, T_frames, hidden_dim]
        last         = out[:, -1, :]                    # [B, hidden_dim]
        log_scale    = self.head(last)                  # [B, 1]
        return log_scale, hidden

    def reset_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero hidden state for the start of a new audio sequence."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )


# ---------------------------------------------------------------------------
# 2.  Chunking helpers  (mirrors EnCodec 48 kHz internals)
# ---------------------------------------------------------------------------

def get_chunks(wav: torch.Tensor) -> List[torch.Tensor]:
    """
    Split [B, C, T] into a list of [B, C, CHUNK_SAMPLES] tensors using
    the same stride / overlap EnCodec 48 kHz uses internally.

    The final short chunk is zero-padded to CHUNK_SAMPLES so that the
    SEANet CNN strides always see a full-length input.
    """
    T, chunks, start = wav.shape[-1], [], 0
    while start < T:
        end   = min(start + CHUNK_SAMPLES, T)
        chunk = wav[:, :, start:end]
        if chunk.shape[-1] < CHUNK_SAMPLES:
            chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[-1]))
        chunks.append(chunk)
        start += STRIDE_SAMPLES
    return chunks


def get_gt_scales(
    model: EncodecModel, wav: torch.Tensor
) -> List[torch.Tensor]:
    """
    Run the standard EnCodec encode and return only the ground-truth scales.
    Returns a list of [B, 1] tensors, one per chunk.
    """
    with torch.no_grad():
        encoded_frames = model.encode(wav)      # list of (codes, scale)
    return [scale for _, scale in encoded_frames]


def get_encoder_latents(
    model: EncodecModel, wav: torch.Tensor
) -> List[torch.Tensor]:
    """
    Run model.encoder (the SEANet encoder) on each chunk and return a list
    of [B, HIDDEN_SIZE, T_frames] tensors, one per chunk.

    Calling model.encoder directly gives us the continuous latent
    representation before quantization — this is what the GRU reads.
    """
    latents = []
    with torch.no_grad():
        for chunk in get_chunks(wav):
            z = model.encoder(chunk)            # [B, 128, T_frames]
            latents.append(z)
    return latents


# ---------------------------------------------------------------------------
# 3.  Inference wrapper
# ---------------------------------------------------------------------------

class LearnedScaleCodec:
    """
    Wraps EncodecModel (48 kHz) and replaces the hard per-chunk scale with
    the output of a trained ScalePredictor.

    The predictor is causal: for chunk t it only uses information from encoder
    latents of chunks 0 … t, never peeking at future audio.

    Parameters
    ----------
    model         : EncodecModel  (encodec_model_48khz, already .eval())
    predictor     : trained ScalePredictor
    use_predicted : if False, fall back to ground-truth scale for ablation
    """

    def __init__(
        self,
        model: EncodecModel,
        predictor: ScalePredictor,
        use_predicted: bool = True,
    ):
        self.model         = model
        self.predictor     = predictor
        self.use_predicted = use_predicted
        self.predictor.eval()

    @torch.no_grad()
    def encode(
        self, wav: torch.Tensor
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Encode wav and return a list of (codes, scale) tuples — the same
        format as model.encode() — but with scales replaced by predicted ones.

        Parameters
        ----------
        wav : [B, C, T]  audio already at 48 kHz stereo

        Returns
        -------
        List of (codes [B, n_q, frame_len], scale [B, 1])
        """
        encoded_frames = self.model.encode(wav)   # [(codes, gt_scale), ...]

        if not self.use_predicted:
            return encoded_frames

        predicted_scales = self._predict_scales(wav)
        n = min(len(encoded_frames), len(predicted_scales))
        return [(encoded_frames[i][0], predicted_scales[i]) for i in range(n)]

    @torch.no_grad()
    def decode(
        self,
        encoded_frames: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> torch.Tensor:
        """Decode (codes, scale) frames back to a waveform via model.decode()."""
        return self.model.decode(encoded_frames)

    @torch.no_grad()
    def encode_decode(self, wav: torch.Tensor) -> torch.Tensor:
        """Full round-trip with predicted scales."""
        return self.decode(self.encode(wav))

    # ------------------------------------------------------------------
    # Streaming: chunk encode with GRU-predicted scale
    # ------------------------------------------------------------------

    def encode_chunks_predicted(
        self,
        wav: torch.Tensor,
        chunk_samples: int,
        hop_samples: int,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Slide a window over wav [1, C, T], encode each chunk, and replace
        its scale with the GRU-predicted scale.

        The GRU hidden state is carried chunk-to-chunk so the predictor
        has causal memory of the loudness trajectory.

        Returns a list of (codes, predicted_scale) frames in the same
        format as ``encode_chunks`` from src.overlap_add, ready to pass
        directly to ``overlap_add_decode``.

        Parameters
        ----------
        wav           : [1, C, T]  input waveform
        chunk_samples : window length in samples
        hop_samples   : step between consecutive windows in samples

        Returns
        -------
        List of (codes [B, K, T_frames], predicted_scale [B, 1])
        """
        total_samples = wav.shape[-1]
        frames: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
        hidden = self.predictor.reset_hidden(wav.shape[0], wav.device)

        start = 0
        while start < total_samples:
            end   = min(start + chunk_samples, total_samples)
            chunk = wav[:, :, start:end]
            if chunk.shape[-1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

            with torch.no_grad():
                chunk_frames      = self.model.encode(chunk)   # [(codes, gt_scale)]
                codes             = chunk_frames[0][0]
                z                 = self.model.encoder(chunk)  # [B, HIDDEN_SIZE, T_frames]
                log_scale, hidden = self.predictor(z, hidden)
                pred_scale        = torch.exp(log_scale)

            frames.append((codes, pred_scale))
            start += hop_samples

        return frames

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_scales(self, wav: torch.Tensor) -> List[torch.Tensor]:
        """
        Run encoder chunk-by-chunk → feed each latent through the GRU →
        return a list of predicted scale tensors [B, 1].
        """
        latents = get_encoder_latents(self.model, wav)
        B, device = wav.shape[0], wav.device
        hidden    = self.predictor.reset_hidden(B, device)

        scales = []
        for z in latents:
            log_scale, hidden = self.predictor(z, hidden)
            scales.append(torch.exp(log_scale))      # log → linear
        return scales

    # Reuse src.audio.load_audio as a method (signature: (self, path) → [1, C, T])
    load_audio = load_audio


# ---------------------------------------------------------------------------
# 4.  Dataset
# ---------------------------------------------------------------------------

class AudioChunkDataset(Dataset):
    """
    Scans a directory for .wav files and returns fixed-length clips for
    training the ScalePredictor.

    Each item is a [2, clip_samples] float32 tensor at 48 kHz stereo.
    Use clip_seconds >= 3 so the GRU sees at least a few chunks per forward pass.
    """

    def __init__(self, audio_dir: str, clip_seconds: float = 5.0):
        self.clip_len = int(clip_seconds * SAMPLE_RATE)
        self.clips: List[Tuple[str, int]] = []

        for fname in sorted(os.listdir(audio_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(audio_dir, fname)
            info  = sf.info(fpath)
            for i in range(info.frames // self.clip_len):
                self.clips.append((fpath, i * self.clip_len))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fpath, offset = self.clips[idx]
        wav, sr = torchaudio.load(fpath, frame_offset=offset, num_frames=self.clip_len)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] == 1:
            wav = wav.expand(2, -1).clone()
        elif wav.shape[0] > 2:
            wav = wav[:2]
        return wav                                    # [2, clip_len]


# ---------------------------------------------------------------------------
# 5.  Training
# ---------------------------------------------------------------------------

def train_predictor(
    audio_dir: str,
    save_path: str = "scale_predictor.pt",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    device_str: str = "cpu",
) -> ScalePredictor:
    """
    Train the ScalePredictor on a folder of stereo 48 kHz .wav files.
    EnCodec is kept frozen — only the predictor's parameters are updated.

    Loss: MSE in log-scale space  →  scale-invariant across loud/quiet passages.

        L = MSE( log_pred,  log(scale_gt) )
    """
    device = torch.device(device_str)

    print("Loading EnCodec 48 kHz …")
    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(6.0)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    predictor = ScalePredictor(input_dim=HIDDEN_SIZE, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    dataset = AudioChunkDataset(audio_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} clips  |  {epochs} epochs  |  device: {device}\n")

    for epoch in range(1, epochs + 1):
        predictor.train()
        total_loss = 0.0

        for batch_wav in loader:
            batch_wav = batch_wav.to(device)           # [B, 2, clip_len]

            gt_scales = get_gt_scales(model, batch_wav)
            latents   = get_encoder_latents(model, batch_wav)

            # Align chunk counts (overlap can cause ±1 difference)
            n         = min(len(latents), len(gt_scales))
            latents   = latents[:n]
            gt_scales = gt_scales[:n]

            B      = batch_wav.shape[0]
            hidden = predictor.reset_hidden(B, device)
            chunk_losses = []

            for z, gt_s in zip(latents, gt_scales):
                log_pred, hidden = predictor(z, hidden)
                log_gt           = torch.log(gt_s.clamp(min=1e-8))
                chunk_losses.append(loss_fn(log_pred, log_gt))

            loss = torch.stack(chunk_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch:3d}/{epochs}  avg MSE loss: {avg:.6f}")

    torch.save({"predictor": predictor.state_dict(), "hidden_dim": hidden_dim}, save_path)
    print(f"\nPredictor saved → {save_path}")
    return predictor


# ---------------------------------------------------------------------------
# 6.  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learned causal GRU scale predictor for EnCodec — OLA streaming demo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── eval ──────────────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser(
        "eval", help="Compare OLA reconstructions with and without predicted scale."
    )
    eval_p.add_argument("input_audio", type=Path, help="Input audio file path.")
    eval_p.add_argument("predictor",   type=Path, help="Path to trained predictor checkpoint (.pt).")
    eval_p.add_argument(
        "--codec", type=str, default="encodec48", choices=["encodec48", "encodec24"],
        help="EnCodec model variant.",
    )
    eval_p.add_argument("--bandwidth",     type=float, default=6.0,  help="Encodec bandwidth in kbps.")
    eval_p.add_argument("--chunk-seconds", type=float, default=1.0,  help="Streaming chunk size in seconds.")
    eval_p.add_argument("--hop-seconds",   type=float, default=0.5,  help="Streaming hop size in seconds.")
    eval_p.add_argument(
        "--window", type=str, default="hann", choices=["hann", "rect"],
        help="Synthesis window for overlap-add reconstruction.",
    )
    eval_p.add_argument(
        "--device", type=str, default="auto", help="Device to use: auto, cpu, cuda, or mps."
    )

    # ── train ─────────────────────────────────────────────────────────────────
    train_p = subparsers.add_parser(
        "train", help="Train the ScalePredictor on a folder of .wav files."
    )
    train_p.add_argument("audio_dir",  type=Path, help="Directory containing stereo 48 kHz .wav files.")
    train_p.add_argument(
        "save_path", type=Path, nargs="?", default=Path("scale_predictor.pt"),
        help="Output checkpoint path (default: scale_predictor.pt).",
    )
    train_p.add_argument("--epochs",     type=int,   default=20,   help="Training epochs.")
    train_p.add_argument("--batch-size", type=int,   default=4,    help="Batch size.")
    train_p.add_argument("--lr",         type=float, default=1e-3, help="Learning rate.")
    train_p.add_argument("--hidden-dim", type=int,   default=64,   help="GRU hidden size.")
    train_p.add_argument(
        "--device", type=str, default="auto", help="Device to use: auto, cpu, cuda, or mps."
    )

    # ── demo ──────────────────────────────────────────────────────────────────
    subparsers.add_parser("demo", help="Architecture smoke test (no audio or checkpoint needed).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── demo ──────────────────────────────────────────────────────────────────
    if args.command == "demo":
        print("Running architecture smoke test …\n")
        predictor = ScalePredictor(input_dim=HIDDEN_SIZE, hidden_dim=64)
        n_params  = sum(p.numel() for p in predictor.parameters())
        print(f"ScalePredictor parameters: {n_params:,}")

        hidden = None
        for i in range(2):
            z_fake        = torch.randn(2, HIDDEN_SIZE, 150)   # [B=2, 128, 150]
            log_s, hidden = predictor(z_fake, hidden)
            scale         = torch.exp(log_s)
            print(f"  Chunk {i}  latent {list(z_fake.shape)}  →  scale {list(scale.shape)}  {scale.squeeze().tolist()}")

        print("\nSmoke test passed ✓")
        return

    # ── train ─────────────────────────────────────────────────────────────────
    if args.command == "train":
        device_str = get_device() if args.device == "auto" else args.device
        train_predictor(
            str(args.audio_dir),
            save_path=str(args.save_path),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            device_str=device_str,
        )
        return

    # ── eval ──────────────────────────────────────────────────────────────────
    device = torch.device(get_device() if args.device == "auto" else args.device)

    # Load EnCodec model
    if args.codec == "encodec48":
        model = EncodecModel.encodec_model_48khz()
    else:
        model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(args.bandwidth)
    model.to(device).eval()

    # Load predictor checkpoint
    ckpt      = torch.load(str(args.predictor), map_location=device)
    predictor = ScalePredictor(input_dim=HIDDEN_SIZE, hidden_dim=ckpt["hidden_dim"]).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    codec = LearnedScaleCodec(model, predictor)

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
        full_frames = model.encode(wav)
        full_recon  = model.decode(full_frames)

    # ── streaming OLA: raw per-chunk scale ────────────────────────────────────
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

    # ── streaming OLA: GRU-predicted scale ────────────────────────────────────
    pred_frames = codec.encode_chunks_predicted(wav, chunk_samples, hop_samples)
    predicted_recon = overlap_add_decode(
        model, pred_frames,
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
        evaluate_pair(wav, predicted_recon, label="Predicted-scale OLA"),
    ]
    print_results(results, reference_label="Full reconstruct")

    # ── save outputs ──────────────────────────────────────────────────────────
    stem          = args.input_audio.stem
    full_output   = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_full.wav"
    stream_output = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_stream.wav"
    pred_output   = Path.cwd() / f"{stem}_{args.codec}_bw{args.bandwidth:g}_pred_scale.wav"

    def _save(path: Path, audio: torch.Tensor) -> None:
        wav_2d = audio.squeeze(0).detach().cpu().float()
        torchaudio.save(str(path), wav_2d, model.sample_rate)

    _save(full_output,   full_recon)
    _save(stream_output, streaming_recon)
    _save(pred_output,   predicted_recon)

    print(f"Saved full reconstruction          → {full_output}")
    print(f"Saved streaming OLA                → {stream_output}")
    print(f"Saved predicted-scale OLA          → {pred_output}")


if __name__ == "__main__":
    main()