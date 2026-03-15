from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from encodec import EncodecModel
import dac


class CodecAdapter:
    name: str
    sample_rate: int
    channels: int

    def full_reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def chunk_reconstruct(
        self, wav: torch.Tensor, scale_mode: str = "per_chunk"
    ) -> torch.Tensor:
        raise NotImplementedError


class EncodecAdapter(CodecAdapter):
    def __init__(self, variant: str, bandwidth: float, device: torch.device) -> None:
        self.variant = variant
        self.name = f"encodec{variant}"
        self.device = device

        if variant == "48":
            self.model = EncodecModel.encodec_model_48khz()
        elif variant == "24":
            self.model = EncodecModel.encodec_model_24khz()
        else:
            raise ValueError("Encodec variant must be '24' or '48'.")

        self.model.set_target_bandwidth(float(bandwidth))
        self.model.to(self.device).eval()
        self.sample_rate = int(self.model.sample_rate)
        self.channels = int(self.model.channels)

    @torch.no_grad()
    def full_reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        frames = self.model.encode(wav)
        rec = self.model.decode(frames)
        return rec

    @torch.no_grad()
    def chunk_reconstruct(
        self, wav: torch.Tensor, scale_mode: str = "per_chunk"
    ) -> torch.Tensor:
        frames = self.model.encode(wav)
        if self.variant == "48" and scale_mode == "first_chunk":
            # Reuse the first chunk's scale for all frames to reduce chunk-to-chunk gain drift.
            first_scale = self._extract_first_scale(frames)
            frames = self._replace_scales(frames, first_scale)
        rec = self.model.decode(frames)
        return rec

    def _extract_first_scale(self, frames: Sequence) -> Optional[torch.Tensor]:
        for frame in frames:
            if isinstance(frame, tuple) and len(frame) == 2:
                _, scale = frame
                if scale is not None:
                    return scale.detach().clone()
        return None

    def _replace_scales(
        self, frames: Sequence, scale_ref: Optional[torch.Tensor]
    ) -> List:
        if scale_ref is None:
            return list(frames)

        replaced = []
        for frame in frames:
            if isinstance(frame, tuple) and len(frame) == 2:
                codes, scale = frame
                if scale is None:
                    replaced.append((codes, None))
                else:
                    s = scale_ref
                    if s.shape != scale.shape:
                        try:
                            # Broadcast when Encodec emits compatible but non-identical shapes.
                            s = s.expand_as(scale)
                        except RuntimeError:
                            # Last-resort fallback: keep mean level while matching target shape.
                            s = torch.ones_like(scale) * torch.mean(scale_ref)
                    replaced.append((codes, s))
            else:
                replaced.append(frame)
        return replaced


class DACAdapter(CodecAdapter):
    def __init__(self, model_type: str, device: torch.device) -> None:

        self.name = "dac"
        self.device = device

        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        self.model.to(self.device).eval()

        self.sample_rate = int(getattr(self.model, "sample_rate", 44100))
        self.channels = int(getattr(self.model, "audio_channels", 1))

    @torch.no_grad()
    def full_reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        z, *_ = self.model.encode(wav)
        rec = self.model.decode(z)
        return rec

    @torch.no_grad()
    def chunk_reconstruct(
        self, wav: torch.Tensor, scale_mode: str = "per_chunk"
    ) -> torch.Tensor:
        _ = scale_mode
        z, *_ = self.model.encode(wav)
        rec = self.model.decode(z)
        return rec


def make_adapter(
    codec: str, bandwidth: float, device: torch.device, dac_model_type: str
) -> CodecAdapter:
    """Create and return a codec adapter for the requested backend.

    Args:
        codec: Codec identifier. Supported values are ``"encodec48"``,
            ``"encodec24"``, and ``"dac"``.
        bandwidth: Target bandwidth for Encodec adapters. Ignored when
            ``codec == "dac"``.
        device: Torch device where the codec model is loaded.
        dac_model_type: DAC model identifier passed to
            ``dac.utils.download(model_type=...)`` when ``codec == "dac"``.

    Returns:
        A configured ``CodecAdapter`` instance for the requested codec.

    Raises:
        ValueError: If ``codec`` is not one of the supported identifiers.
    """
    if codec == "encodec48":
        return EncodecAdapter("48", bandwidth=bandwidth, device=device)
    if codec == "encodec24":
        return EncodecAdapter("24", bandwidth=bandwidth, device=device)
    if codec == "dac":
        return DACAdapter(model_type=dac_model_type, device=device)
    raise ValueError(f"Unsupported codec: {codec}")
