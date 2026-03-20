"""
run_experiments.py
==================
Runs the full 3 × 3 experiment grid across a dataset folder and saves results.

Grid
----
  Chunk sizes  : 1.0s  0.5s  0.25s
  Strategies   : raw scale  |  EMA smoothing  |  GRU predictor

For every audio file found recursively under --data-dir, each condition is
evaluated and all four metrics (SI-SDR, SNR, L1, MSE) are recorded.
Results are averaged across files and written to a CSV + printed as a table.

Folder structure expected
-------------------------
    data/
      train/
        *.wav
      test/
        *.wav
      000/
        *.flac / *.wav / *.mp3

All audio formats supported by torchaudio are accepted.
The --split argument filters to a single sub-folder (e.g. "test").

Usage
-----
    # Full grid, test split, using a trained GRU predictor:
    python run_experiments.py \\
        --data-dir  data/ \\
        --split     test \\
        --predictor scale_predictor.pt \\
        --bandwidth 6.0 \\
        --max-files 20

    # Without a trained predictor (skips GRU rows):
    python run_experiments.py \\
        --data-dir data/ --split test --no-gru

    # Alpha sweep only (no GRU):
    python run_experiments.py \\
        --data-dir data/ --split test --no-gru \\
        --alphas 1.0 0.7 0.5 0.3 0.1

Output
------
    results/results.csv        — one row per (file, chunk_size, strategy)
    results/summary_table.txt  — averaged metrics, ready to paste into the paper
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from torchaudio.transforms import Resample

from src.device import get_device
from src.eval import evaluate_pair, print_results
from src.overlap_add import encode_chunks, overlap_add_decode

# Import smoothing and GRU classes from your existing scripts
from smooth import SmoothedScaleCodec
from learned_scale_predictor import ScalePredictor, LearnedScaleCodec


# ---------------------------------------------------------------------------
# Audio extensions accepted
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_audio_files(
    data_dir: Path,
    split: Optional[str],
    max_files: Optional[int],
) -> List[Path]:
    """
    Recursively find all audio files under data_dir.
    If split is given, only search inside data_dir / split.
    """
    root = data_dir / split if split else data_dir
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No audio files found under {root}")

    if max_files:
        files = files[:max_files]

    print(f"Found {len(files)} audio files under {root}")
    return files


def load_wav(path: Path, model: EncodecModel, device: torch.device) -> torch.Tensor:
    """Load, resample, channel-match, and move to device. Returns [1, C, T]."""
    wav, sr = torchaudio.load(str(path))
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    return wav.unsqueeze(0).to(device)


def load_model(codec: str, bandwidth: float, device: torch.device) -> EncodecModel:
    if codec == "encodec48":
        model = EncodecModel.encodec_model_48khz()
    else:
        model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    return model.to(device).eval()


def load_predictor(
    path: Path, device: torch.device
) -> ScalePredictor:
    ckpt      = torch.load(str(path), map_location=device)
    predictor = ScalePredictor(input_dim=128, hidden_dim=ckpt["hidden_dim"]).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()
    return predictor


# ---------------------------------------------------------------------------
# Single-file evaluation: all conditions at one chunk size
# ---------------------------------------------------------------------------

def _save_recon(
    wav: torch.Tensor,
    sample_rate: int,
    out_path: Path,
) -> None:
    """Save a [1, C, T] or [C, T] waveform tensor as a wav file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = wav.squeeze(0).cpu()   # [C, T]
    torchaudio.save(str(out_path), w, sample_rate)


def evaluate_file(
    wav: torch.Tensor,
    model: EncodecModel,
    chunk_seconds: float,
    hop_ratio: float,
    alphas: List[float],
    predictor: Optional[ScalePredictor],
    window_type: str,
    audio_out_dir: Optional[Path] = None,
    stem: str = "audio",
) -> List[Dict]:
    """
    Run all scale strategies for a single chunk_seconds value on one file.

    Returns a list of result dicts, one per (chunk_seconds, strategy, alpha).
    Each dict contains: chunk_seconds, strategy, alpha, si_sdr, snr_db, l1_error, mse.

    If audio_out_dir is given, each reconstruction is saved as a wav file there.
    """
    device        = wav.device
    total_samples = wav.shape[-1]
    chunk_samples = max(1, int(round(chunk_seconds * model.sample_rate)))
    hop_samples   = max(1, int(round(chunk_seconds * hop_ratio * model.sample_rate)))
    hop_samples   = min(hop_samples, chunk_samples)

    rows = []
    chunk_tag = f"chunk{chunk_seconds:.2f}s"
    sr = model.sample_rate

    # ── encode chunks once — reused by all scale strategies ──────────────────
    with torch.no_grad():
        chunk_frames = encode_chunks(model, wav, chunk_samples, hop_samples)

    def _ola(frames):
        return overlap_add_decode(
            model, frames,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            total_samples=total_samples,
            channels=model.channels,
            window_type=window_type,
        )

    def _maybe_save(recon: torch.Tensor, label: str) -> None:
        if audio_out_dir is not None:
            out = audio_out_dir / f"{stem}__{chunk_tag}__{label}.wav"
            _save_recon(recon, sr, out)

    # ── 1. Raw scale (no mitigation) ─────────────────────────────────────────
    recon  = _ola(chunk_frames)
    result = evaluate_pair(wav, recon, label="raw")
    rows.append(_row(chunk_seconds, "raw", alpha=None, result=result))
    _maybe_save(recon, "raw")

    # ── 2. EMA smoothing — linear and log, each alpha ─────────────────────────
    for alpha in alphas:
        for mode in ("linear", "log"):
            codec    = SmoothedScaleCodec(model, alpha=alpha, mode=mode)
            smoothed = codec.smooth_frames(chunk_frames)
            recon    = _ola(smoothed)
            result   = evaluate_pair(wav, recon, label=f"ema_{mode}_a{alpha:.2f}")
            rows.append(_row(chunk_seconds, f"ema_{mode}", alpha=alpha, result=result))
            _maybe_save(recon, f"ema_{mode}_a{alpha:.2f}")

    # ── 3. GRU predictor ─────────────────────────────────────────────────────
    if predictor is not None:
        lsc         = LearnedScaleCodec(model, predictor, use_predicted=True)
        pred_frames = lsc.encode_chunks_predicted(wav, chunk_samples, hop_samples)
        recon       = _ola(pred_frames)
        result      = evaluate_pair(wav, recon, label="gru")
        rows.append(_row(chunk_seconds, "gru", alpha=None, result=result))
        _maybe_save(recon, "gru")

    return rows


def _row(
    chunk_seconds: float,
    strategy: str,
    alpha: Optional[float],
    result: Dict,
) -> Dict:
    return {
        "chunk_seconds": chunk_seconds,
        "strategy"     : strategy,
        "alpha"        : alpha if alpha is not None else "",
        "si_sdr"       : result["si_sdr"],
        "snr_db"       : result["snr_db"],
        "l1_error"     : result["l1_error"],
        "mse"          : result["mse"],
    }


# ---------------------------------------------------------------------------
# Summary: average metrics across files, grouped by (chunk_seconds, strategy, alpha)
# ---------------------------------------------------------------------------

def summarise(all_rows: List[Dict]) -> Dict[tuple, Dict]:
    """
    Average all metrics over files, keyed by (chunk_seconds, strategy, alpha).
    Returns an ordered dict suitable for printing.
    """
    from collections import defaultdict

    accum: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in all_rows:
        key = (row["chunk_seconds"], row["strategy"], row["alpha"])
        accum[key].append(row)

    summary = {}
    for key, rows in accum.items():
        n = len(rows)
        summary[key] = {
            "chunk_seconds": key[0],
            "strategy"     : key[1],
            "alpha"        : key[2],
            "n_files"      : n,
            "si_sdr"       : sum(r["si_sdr"]    for r in rows) / n,
            "snr_db"       : sum(r["snr_db"]    for r in rows) / n,
            "l1_error"     : sum(r["l1_error"]  for r in rows) / n,
            "mse"          : sum(r["mse"]        for r in rows) / n,
        }
    return summary


def print_summary_table(summary: Dict[tuple, Dict]) -> str:
    """Print and return the summary table as a string."""
    col = 28
    header = (
        f"{'Condition':<{col}}"
        f"{'SI-SDR↑ (dB)':>13}"
        f"{'SNR↑ (dB)':>11}"
        f"{'L1 error↓':>12}"
        f"{'MSE↓':>14}"
        f"{'N':>5}"
    )
    sep   = "─" * len(header)
    lines = [sep, header, sep]

    prev_chunk = None
    for key, row in summary.items():
        chunk = row["chunk_seconds"]
        # Section separator between chunk sizes
        if chunk != prev_chunk:
            if prev_chunk is not None:
                lines.append("")
            lines.append(f"  Chunk size: {chunk:.2f}s")
            prev_chunk = chunk

        alpha_str = f" α={row['alpha']:.2f}" if row["alpha"] != "" else ""
        label     = f"    {row['strategy']}{alpha_str}"
        lines.append(
            f"{label:<{col}}"
            f"{row['si_sdr']:>13.2f}"
            f"{row['snr_db']:>11.2f}"
            f"{row['l1_error']:>12.6f}"
            f"{row['mse']:>14.8f}"
            f"{row['n_files']:>5}"
        )

    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "file", "chunk_seconds", "strategy", "alpha",
    "si_sdr", "snr_db", "l1_error", "mse",
]

def write_csv(path: Path, file_rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(file_rows)
    print(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full streaming experiment grid across a dataset folder."
    )
    p.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root folder of audio files.",
    )
    p.add_argument(
        "--split", type=str, default=None,
        help="Sub-folder to use, e.g. 'test' or '000'. Omit to use all files.",
    )
    p.add_argument(
        "--predictor", type=Path, default=None,
        help="Path to trained ScalePredictor checkpoint (.pt). "
             "Omit or use --no-gru to skip the GRU condition.",
    )
    p.add_argument(
        "--no-gru", action="store_true",
        help="Skip the GRU predictor condition entirely.",
    )
    p.add_argument(
        "--codec", type=str, default="encodec48",
        choices=["encodec48", "encodec24"],
    )
    p.add_argument(
        "--bandwidth", type=float, default=6.0,
        help="EnCodec target bandwidth in kbps.",
    )
    p.add_argument(
        "--chunk-sizes", type=float, nargs="+", default=[1.0, 0.5, 0.25],
        metavar="SECONDS",
        help="Chunk sizes to evaluate (default: 1.0 0.5 0.25).",
    )
    p.add_argument(
        "--hop-ratio", type=float, default=0.5,
        help="Hop as a fraction of chunk size (default: 0.5 = 50%% overlap).",
    )
    p.add_argument(
        "--alphas", type=float, nargs="+", default=[0.3],
        metavar="ALPHA",
        help="EMA alpha values to sweep (default: 0.3).",
    )
    p.add_argument(
        "--window", type=str, default="hann", choices=["hann", "rect"],
    )
    p.add_argument(
        "--max-files", type=int, default=None,
        help="Cap the number of files (useful for quick runs).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("results"),
        help="Directory for CSV and summary table output (default: results/).",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, cuda, or mps.",
    )
    p.add_argument(
        "--save-audio", action="store_true",
        help="Save every reconstruction as a wav file under <out-dir>/audio/.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device(get_device() if args.device == "auto" else args.device)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading {args.codec} at {args.bandwidth} kbps …")
    model = load_model(args.codec, args.bandwidth, device)

    # ── load predictor (optional) ─────────────────────────────────────────────
    predictor = None
    if not args.no_gru:
        if args.predictor and args.predictor.exists():
            print(f"Loading ScalePredictor from {args.predictor} …")
            predictor = load_predictor(args.predictor, device)
        else:
            print("No --predictor provided — GRU condition will be skipped.")

    # ── find files ────────────────────────────────────────────────────────────
    files = find_audio_files(args.data_dir, args.split, args.max_files)

    # ── run grid ──────────────────────────────────────────────────────────────
    all_rows: List[Dict] = []
    n_total = len(files) * len(args.chunk_sizes)
    done    = 0

    for audio_path in files:
        try:
            wav = load_wav(audio_path, model, device)
        except Exception as e:
            print(f"  [skip] {audio_path.name}: {e}")
            continue

        for chunk_seconds in args.chunk_sizes:
            done += 1
            print(
                f"[{done}/{n_total}] {audio_path.name}"
                f"  chunk={chunk_seconds:.2f}s …",
                end="  ", flush=True,
            )

            audio_out_dir = (
                args.out_dir / "audio" if args.save_audio else None
            )

            try:
                rows = evaluate_file(
                    wav           = wav,
                    model         = model,
                    chunk_seconds = chunk_seconds,
                    hop_ratio     = args.hop_ratio,
                    alphas        = args.alphas,
                    predictor     = predictor,
                    window_type   = args.window,
                    audio_out_dir = audio_out_dir,
                    stem          = audio_path.stem,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            # Attach filename to each row for the CSV
            file_rows = [{"file": audio_path.name, **r} for r in rows]
            all_rows.extend(file_rows)

            # Print a compact per-file SI-SDR summary
            si_sdrs = "  ".join(
                f"{r['strategy'][:8]}={r['si_sdr']:.1f}" for r in rows
            )
            print(si_sdrs)

    if not all_rows:
        print("No results collected — check your data directory and file paths.")
        return

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = args.out_dir / "results.csv"
    write_csv(csv_path, all_rows)

    # ── print + save summary table ────────────────────────────────────────────
    summary = summarise(all_rows)
    print(f"\n{'='*70}")
    print("SUMMARY  (averaged across all files)")
    print(f"{'='*70}\n")
    table_str = print_summary_table(summary)

    table_path = args.out_dir / "summary_table.txt"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table_str)
    print(f"\nSummary table saved → {table_path}")

    # ── save full summary as JSON too (easy to load in a notebook) ────────────
    json_path = args.out_dir / "summary.json"
    json_path.write_text(
        json.dumps(
            {str(k): v for k, v in summary.items()}, indent=2
        )
    )
    print(f"Summary JSON saved  → {json_path}")


if __name__ == "__main__":
    main()
