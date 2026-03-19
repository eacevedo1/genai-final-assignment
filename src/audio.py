"""
Audio I/O and waveform utilities.

Helpers
-------
  load_audio          : Load and resample an audio file to the model's target rate.
  _trim_to_same_length: Trim two tensors to a common length along the last axis.
"""

import torch
import torchaudio
from encodec.utils import convert_audio


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_audio(self, path: str) -> torch.Tensor:
    """
    Load an audio file from disk and resample it to the model's sample rate
    and channel count.

    Parameters
    ----------
    path : str
        Path to the audio file (any format supported by torchaudio).

    Returns
    -------
    torch.Tensor  shape [1, C, T]
        Waveform with a leading batch dimension added.
    """
    wav, sr = torchaudio.load(path)
    wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
    return wav.unsqueeze(0)  # add batch dim → [1, C, T]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trim_to_same_length(
    ref: torch.Tensor, est: torch.Tensor
) -> tuple:
    """
    Trim both tensors to the shorter length along the last dimension.

    Parameters
    ----------
    ref : torch.Tensor   reference waveform  [..., T]
    est : torch.Tensor   estimate waveform   [..., T]

    Returns
    -------
    tuple of (ref[..., :n], est[..., :n])  where n = min(T_ref, T_est)
    """
    n = min(ref.shape[-1], est.shape[-1])
    return ref[..., :n], est[..., :n]