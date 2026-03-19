"""
Evaluation metrics for reconstructed waveforms.
Metrics
-------
  si_sdr   : Scale-Invariant Signal-to-Distortion Ratio  (dB, higher is better)
  snr_db   : Signal-to-Noise Ratio                       (dB, higher is better)
  l1_error : Mean Absolute Error between waveforms       (lower is better)
  mse      : Mean Squared Error between waveforms        (lower is better)

All metrics operate on raw waveform tensors — no perceptual weighting.
"""

import sys
from typing import Dict, Optional

import torch
import torchaudio
from .audio import _trim_to_same_length, load_audio

from encodec.utils import convert_audio

SAMPLE_RATE = 48_000
CHANNELS    = 2


# ---------------------------------------------------------------------------
# Core metric functions
# All accept:  reference [C, T] or [T],  estimate [C, T] or [T]
# All return:  float  (scalar)
# ---------------------------------------------------------------------------

def si_sdr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Measures reconstruction quality independently of any global amplitude
    difference between reference and estimate.  Higher is better.

    SI-SDR = 10 * log10( ||alpha * ref||^2 / ||alpha * ref - est||^2 )
    where alpha = <est, ref> / <ref, ref>  (optimal scale factor)

    Reference: Le Roux et al., ICASSP 2019.
    """
    ref, est = _flatten_and_align(reference, estimate)
    ref = ref - ref.mean()
    est = est - est.mean()

    alpha  = (est @ ref) / (ref @ ref + 1e-8)
    target = alpha * ref
    noise  = est - target

    return _safe_db(target @ target, noise @ noise)


def snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """
    Signal-to-Noise Ratio (SNR) in dB.

    Unlike SI-SDR, SNR does not optimally rescale the estimate — it measures
    the raw error relative to the signal power.  More sensitive to amplitude
    mismatches.  Higher is better.

    SNR = 10 * log10( ||ref||^2 / ||ref - est||^2 )
    """
    ref, est = _flatten_and_align(reference, estimate)
    noise    = ref - est
    return _safe_db(ref @ ref, noise @ noise)


def l1_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """
    Mean Absolute Error (L1) between reference and estimate waveforms.

    L1 = (1/N) * sum |ref_i - est_i|

    Lower is better.  More robust to occasional large errors than MSE.
    """
    ref, est = _flatten_and_align(reference, estimate)
    return (ref - est).abs().mean().item()


def mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """
    Mean Squared Error (MSE) between reference and estimate waveforms.

    MSE = (1/N) * sum (ref_i - est_i)^2

    Lower is better.  Penalises large errors more heavily than L1.
    """
    ref, est = _flatten_and_align(reference, estimate)
    return ((ref - est) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Convenience: evaluate a single (reference, estimate) pair → dict
# ---------------------------------------------------------------------------

def evaluate_pair(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    label: str = "estimate",
) -> Dict[str, float]:
    """
    Compute all four metrics for a single (reference, estimate) pair.

    Parameters
    ----------
    reference : [C, T] or [B, C, T]   ground-truth waveform
    estimate  : [C, T] or [B, C, T]   reconstructed waveform
    label     : name shown in printed output

    Returns
    -------
    dict with keys: label, si_sdr, snr_db, l1_error, mse
    """
    # Squeeze batch dim if present
    ref = reference.squeeze(0).cpu().float()
    est = estimate.squeeze(0).cpu().float()

    return {
        "label"   : label,
        "si_sdr"  : si_sdr(ref, est),
        "snr_db"  : snr_db(ref, est),
        "l1_error": l1_error(ref, est),
        "mse"     : mse(ref, est),
    }


# ---------------------------------------------------------------------------
# Pretty-print a list of result dicts
# ---------------------------------------------------------------------------

def print_results(results: list, reference_label: str = "GT baseline"):
    """
    Print a formatted comparison table.

    Parameters
    ----------
    results        : list of dicts from evaluate_pair()
    reference_label: label of the first entry (used as the gap reference)
    """
    col_w = 22
    header = (
        f"{'Method':<{col_w}}"
        f"{'SI-SDR (dB)':>14}"
        f"{'SNR (dB)':>12}"
        f"{'L1 error':>12}"
        f"{'MSE':>14}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['label']:<{col_w}}"
            f"{r['si_sdr']:>13.2f} "
            f"{r['snr_db']:>11.2f} "
            f"{r['l1_error']:>12.6f}"
            f"{r['mse']:>14.8f}"
        )

    print(sep)

    # Print gaps relative to first result (GT baseline)
    if len(results) > 1:
        ref_r = results[0]
        print(f"\nGaps relative to [{ref_r['label']}]:")
        for r in results[1:]:
            print(
                f"  {r['label']:<{col_w-2}}"
                f"  SI-SDR {r['si_sdr']  - ref_r['si_sdr']:>+7.2f} dB"
                f"  |  SNR {r['snr_db']  - ref_r['snr_db']:>+7.2f} dB"
                f"  |  L1  {r['l1_error']- ref_r['l1_error']:>+.6f}"
                f"  |  MSE {r['mse']     - ref_r['mse']:>+.8f}"
            )
        print()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_and_align(
    reference: torch.Tensor, estimate: torch.Tensor
) -> tuple:
    """Flatten to 1D double tensors of the same length."""
    ref = reference.flatten().double()
    est = estimate.flatten().double()
    n   = min(ref.shape[0], est.shape[0])
    return ref[:n], est[:n]


def _safe_db(signal_power: torch.Tensor, noise_power: torch.Tensor) -> float:
    """Compute 10*log10(signal/noise), clamping noise to avoid -inf."""
    return (10 * torch.log10(signal_power / noise_power.clamp(min=1e-8))).item()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate EnCodec reconstruction quality."
    )
    parser.add_argument(
        "files", nargs="+",
        help=(
            "Two or more wav files. First file is always the reference. "
            "Remaining files are estimates to evaluate against it. "
            "Example: python eval.py ref.wav gru.wav lm_gru.wav"
        ),
    )
    parser.add_argument(
        "--sr", type=int, default=SAMPLE_RATE,
        help=f"Expected sample rate (default: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--mono", action="store_true",
        help="Load as mono (1 channel) instead of stereo",
    )
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Error: provide at least two wav files (reference + one estimate).")
        sys.exit(1)

    channels = 1 if args.mono else CHANNELS

    # Load reference
    ref_path = args.files[0]
    print(f"\nReference : {ref_path}")
    ref_wav   = load_audio(ref_path)

    # Load estimates and evaluate
    results = []
    for est_path in args.files[1:]:
        print(f"Estimate  : {est_path}")
        est_wav = load_audio(est_path)
        ref_t, est_t = _trim_to_same_length(ref_wav, est_wav)

        import os
        label = os.path.splitext(os.path.basename(est_path))[0]
        results.append(evaluate_pair(ref_t, est_t, label=label))

    # Also evaluate reference against itself as sanity check
    ref_self = evaluate_pair(ref_wav, ref_wav, label="GT baseline (self)")
    all_results = [ref_self] + results

    print_results(all_results)