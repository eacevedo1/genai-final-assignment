import math
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
import soundfile as sf


def _match_channels(
    tensor: torch.Tensor,
    target_channels: int,
    *,
    channel_dim: int,
    mix_to_mono: bool = False,
) -> torch.Tensor:
    """Resize a tensor channel dimension by truncating, mixing, or repeating channels."""
    channels = tensor.shape[channel_dim]
    if channels == target_channels:
        return tensor

    if channels > target_channels:
        if mix_to_mono and target_channels == 1:
            return tensor.mean(dim=channel_dim, keepdim=True)

        slices = [slice(None)] * tensor.ndim
        slices[channel_dim] = slice(target_channels)
        return tensor[tuple(slices)]

    repeats = [1] * tensor.ndim
    repeats[channel_dim] = math.ceil(target_channels / channels)
    expanded = tensor.repeat(*repeats)
    slices = [slice(None)] * tensor.ndim
    slices[channel_dim] = slice(target_channels)
    return expanded[tuple(slices)]


def _match_samples_3d(audio: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Resize time dimension of [B, C, T] by cropping or zero-padding."""
    samples = audio.shape[2]
    if samples == target_samples:
        return audio
    if samples > target_samples:
        return audio[:, :, :target_samples]
    return torch.nn.functional.pad(audio, (0, target_samples - samples))


def _to_2d_audio(wav: torch.Tensor) -> torch.Tensor:
    """Normalize input tensor to [C, T] for audio I/O functions."""
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim != 2:
        raise ValueError(f"Expected [B, C, T] or [C, T], got shape {tuple(wav.shape)}")
    return wav


def fit_audio_shape(
    audio: torch.Tensor, target_channels: int, target_samples: int
) -> torch.Tensor:
    """Crop/pad [B, C, T] audio to exactly target channel/time dimensions."""
    audio = _match_channels(audio, target_channels, channel_dim=1)
    return _match_samples_3d(audio, target_samples)


def build_analysis_window(
    length: int, device: torch.device, window_type: str
) -> torch.Tensor:
    if window_type == "rect" or (window_type == "hann" and length <= 1):
        return torch.ones(length, device=device)
    if window_type == "hann":
        return torch.hann_window(length, periodic=False, device=device)
    raise ValueError(f"Unsupported window type: {window_type}")


def safe_save_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    """Save [B, C, T] or [C, T] audio to disk with torchaudio, fallback to soundfile."""
    path.parent.mkdir(parents=True, exist_ok=True)

    wav_cpu = _to_2d_audio(wav).detach().to("cpu").float().contiguous()

    try:
        torchaudio.save(str(path), wav_cpu, sample_rate)
        return
    except (ImportError, RuntimeError, OSError) as torchaudio_error:
        if sf is None:
            raise RuntimeError(
                "Failed to save audio with torchaudio and soundfile is not installed. "
                "Install soundfile (`pip install soundfile`)."
            ) from torchaudio_error

    sf.write(str(path), wav_cpu.transpose(0, 1).numpy(), sample_rate)


def safe_load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio with torchaudio, falling back to soundfile when needed."""
    try:
        wav, sr = torchaudio.load(str(path))  # [C, T]
        return wav, int(sr)
    except (ImportError, RuntimeError, OSError) as torchaudio_error:
        if sf is None:
            raise ImportError(
                "Audio loading failed with torchaudio and soundfile is not installed. "
                "Install soundfile (`pip install soundfile`) or ensure torchaudio dependencies are available."
            ) from torchaudio_error

    data, sr = sf.read(str(path), always_2d=True)
    wav = torch.from_numpy(data.T).float()  # [C, T]
    return wav, int(sr)


def load_and_prepare_audio(
    path: Path,
    target_sr: int,
    target_channels: int,
    device: torch.device,
) -> torch.Tensor:
    wav, sr = safe_load_audio(path)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

    wav = _match_channels(wav, target_channels, channel_dim=0, mix_to_mono=True)

    wav = wav.unsqueeze(0).to(device)  # [B=1, C, T]
    return wav


def iter_audio_files(input_dir: Path, pattern: str) -> Iterable[Path]:
    return iter(sorted(input_dir.rglob(pattern)))
