"""Frozen Whisper encoder feature extractor.

PROJECT_PLAN.md s3.1 architecture step: take a 16 kHz mono waveform, run
the Whisper-small encoder forward (frozen, no grad), return per-layer
hidden states.

Design choices and why:

- Use `WhisperModel` (not `WhisperForConditionalGeneration`) -- we only
  need the encoder, so we skip the decoder weights and save ~60% RAM.
- `use_safetensors=True` is mandatory on torch<2.6 per CVE-2025-32434
  (PROJECT_PLAN.md s9.4). Asserted at load time.
- Encoder weights are frozen via `.requires_grad_(False)` AND
  `.eval()` -- the second is also load-bearing because Whisper has
  layer norms with running stats that drift in train mode.
- Output is a stacked Tensor[L+1, T, D] of all encoder layer outputs
  (12 layers + the embedding output = 13 entries for Whisper-small),
  not a per-layer dict. The stacked tensor enables a single learnable
  alpha vector and one matmul for the layer-weighted sum in M3.
- All extraction runs in `torch.inference_mode()` to disable autograd
  bookkeeping entirely (faster than `no_grad()`).

Not in this file:

- The trainable layer-weighting head (lands in src/natscore/model.py for M3).
- Attention pooling (src/natscore/pooling.py for M3).
- Disk caching (src/natscore/data/feature_cache.py).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# `transformers` and `soundfile` are runtime deps; import at top so missing-dep
# errors surface early rather than mid-extraction.
import soundfile as sf
from transformers import WhisperFeatureExtractor as _HFWhisperFeatureExtractor
from transformers import WhisperModel

DEFAULT_MODEL = "openai/whisper-small"
WHISPER_SR = 16_000          # Hz; Whisper is hard-coded to 16 kHz
WHISPER_FRAME_RATE = 50      # encoder output frames per second of audio
WHISPER_MAX_AUDIO_SEC = 30   # encoder is fixed to 3000 input mel frames

AudioInput = bytes | bytearray | memoryview | np.ndarray | torch.Tensor | str | Path


@dataclass(frozen=True)
class WhisperFeatureMeta:
    """Static metadata about the loaded encoder. Useful for cache schemas."""

    model_name: str
    n_layers: int                 # number of transformer layers
    n_hidden_states: int          # = n_layers + 1 (embedding + each layer)
    hidden_dim: int
    sample_rate: int
    frame_rate: int               # encoder output frames per second
    max_audio_seconds: int
    output_frames: int            # = max_audio_seconds * frame_rate

    def bytes_per_clip(self, dtype: torch.dtype = torch.float16) -> int:
        bytes_per_value = torch.finfo(dtype).bits // 8
        return self.n_hidden_states * self.output_frames * self.hidden_dim * bytes_per_value


class WhisperFeatureExtractor:
    """Frozen Whisper-small encoder wrapper for naturalness-scoring features."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Per PROJECT_PLAN.md s9.4: torch<2.6 + non-safetensors checkpoint is
        # a known security issue. Force safetensors regardless of torch version.
        self._mel = _HFWhisperFeatureExtractor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name, use_safetensors=True)

        # We only need the encoder. Dropping the decoder halves peak RAM.
        self._encoder = model.get_encoder()
        del model

        self._encoder.requires_grad_(False)
        self._encoder.eval()

        self._device = torch.device(device) if device else torch.device("cpu")
        self._dtype = dtype
        self._encoder.to(self._device, dtype=self._dtype)

        cfg = self._encoder.config
        self._meta = WhisperFeatureMeta(
            model_name=model_name,
            n_layers=cfg.encoder_layers,
            n_hidden_states=cfg.encoder_layers + 1,
            hidden_dim=cfg.d_model,
            sample_rate=WHISPER_SR,
            frame_rate=WHISPER_FRAME_RATE,
            max_audio_seconds=WHISPER_MAX_AUDIO_SEC,
            output_frames=WHISPER_MAX_AUDIO_SEC * WHISPER_FRAME_RATE,
        )

    @property
    def meta(self) -> WhisperFeatureMeta:
        return self._meta

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------ inputs

    def _decode_audio(self, audio: AudioInput) -> np.ndarray:
        """Return a 1-D float32 numpy array at WHISPER_SR Hz."""
        if isinstance(audio, (bytes, bytearray, memoryview)):
            wav, sr = sf.read(io.BytesIO(bytes(audio)), dtype="float32", always_2d=False)
        elif isinstance(audio, (str, Path)):
            wav, sr = sf.read(str(audio), dtype="float32", always_2d=False)
        elif isinstance(audio, np.ndarray):
            wav = audio.astype(np.float32, copy=False)
            sr = WHISPER_SR
        elif isinstance(audio, torch.Tensor):
            wav = audio.detach().to("cpu", torch.float32).numpy()
            sr = WHISPER_SR
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

        # Downmix to mono if needed.
        if wav.ndim == 2:
            wav = wav.mean(axis=1).astype(np.float32, copy=False)
        elif wav.ndim != 1:
            raise ValueError(f"Audio must be 1-D or 2-D; got shape {wav.shape}")

        # Resample if sample rate doesn't match. librosa is the realistic fallback
        # since soundfile does no resampling.
        if sr != WHISPER_SR:
            import librosa  # heavy; lazy-import only when needed

            wav = librosa.resample(wav, orig_sr=sr, target_sr=WHISPER_SR).astype(np.float32, copy=False)

        return wav

    # ----------------------------------------------------------------- extract

    @torch.inference_mode()
    def extract_layerwise(self, audio: AudioInput) -> torch.Tensor:
        """Return Tensor[H, T, D] where H = n_hidden_states, T = output_frames."""
        return self.batch_extract_layerwise([audio])[0]

    @torch.inference_mode()
    def batch_extract_layerwise(self, audios: Sequence[AudioInput]) -> torch.Tensor:
        """Batched extraction. Returns Tensor[B, H, T, D]."""
        if len(audios) == 0:
            raise ValueError("audios is empty")

        wavs = [self._decode_audio(a) for a in audios]
        inputs = self._mel(
            wavs,
            sampling_rate=WHISPER_SR,
            return_tensors="pt",
            padding="max_length",      # always pad to 30s -> 3000 mel frames
        )
        input_features = inputs.input_features.to(self._device, dtype=self._dtype)

        out = self._encoder(input_features, output_hidden_states=True, return_dict=True)
        # `hidden_states` is a tuple of length (n_layers + 1), each Tensor[B, T, D].
        # Stack along dim=1 -> Tensor[B, H, T, D].
        stacked = torch.stack(out.hidden_states, dim=1)
        return stacked

    @torch.inference_mode()
    def extract_pooled(self, audio: AudioInput, layer: int = -1) -> torch.Tensor:
        """Mean-pool across time of a single layer's hidden state. Returns Tensor[D]."""
        feats = self.extract_layerwise(audio)  # [H, T, D]
        return feats[layer].mean(dim=0)

    # ------------------------------------------------------------------ utils

    def __repr__(self) -> str:
        return (
            f"WhisperFeatureExtractor(model_name={self._meta.model_name!r}, "
            f"n_layers={self._meta.n_layers}, hidden_dim={self._meta.hidden_dim}, "
            f"device={self._device}, dtype={self._dtype})"
        )
