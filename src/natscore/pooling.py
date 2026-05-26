"""Attention pooler used by the NatScore head.

PROJECT_PLAN.md s3.1 step 3: a small (768 -> 256 -> 1) attention-score
network that pools a sequence of frame-level hidden states into one
clip-level vector. Trainable params ~200K.

Why attention pooling and not mean-pooling: naturalness signal is not
uniformly distributed in time. Prosodic artifacts cluster at phrase
boundaries; codec glitches happen on rare frames; the front of the
clip often carries the cleanest signal in autoregressive TTS. Mean
pooling washes those out. Attention learns where to look.

Mask handling: Whisper-small always pads its output to 1500 frames
regardless of input length. For training we pass a per-clip
`valid_frames` count derived from duration_seconds * 50 frames/s; the
pooler masks padded frames to -inf before softmax so they contribute
zero weight. Inference can skip the mask and the result is identical
to a clip exactly 30 seconds long (because Whisper itself is mask-free
internally).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionPoolerConfig:
    hidden_dim: int = 768
    bottleneck_dim: int = 256


class AttentionPooler(nn.Module):
    """Soft-attention pooler over the time dimension."""

    def __init__(self, hidden_dim: int = 768, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool [B, T, D] -> [B, D].

        Args:
            x: hidden states, shape [B, T, D].
            valid_frames: optional Long tensor of shape [B] giving the
                number of unpadded frames per item. Frames at index >=
                valid_frames[i] receive zero attention weight.
        """
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, D]; got {tuple(x.shape)}")
        B, T, _ = x.shape

        logits = self.score(x).squeeze(-1)  # [B, T]

        if valid_frames is not None:
            if valid_frames.shape != (B,):
                raise ValueError(
                    f"valid_frames shape {tuple(valid_frames.shape)} != ({B},)"
                )
            arange = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
            mask = arange < valid_frames.unsqueeze(1).to(x.device)  # [B, T]
            logits = logits.masked_fill(~mask, float("-inf"))

        attn = F.softmax(logits, dim=-1)  # [B, T]
        # If a row was all-masked (valid_frames == 0), softmax yields NaNs.
        # Guard with a uniform fallback so loss math stays finite.
        attn = torch.nan_to_num(attn, nan=1.0 / T)

        pooled = torch.einsum("bt,btd->bd", attn, x)
        return pooled
