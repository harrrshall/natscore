"""NatScoreHead: the trainable network that sits on top of frozen Whisper.

PROJECT_PLAN.md s3.1 architecture (post-frozen-encoder stages):

    [B, H, T, D]                            # H = n_hidden_states = 13
        |
        | LayerWeightedSum: softmax(alpha) over H -> [B, T, D]
        v
    [B, T, D]
        |
        | AttentionPooler                   -> [B, D]
        v
    [B, D]
        |
        | ScoreHead MLP (D -> 256 -> 1)     -> [B]
        v
    scalar logit s  (higher = more natural)

Trainable parameter budget (Whisper-small, D=768, H=13):

    LayerWeightedSum:           13 params (just alpha)
    AttentionPooler:            768*256 + 256 + 256*1 + 1     ~= 197 K
    ScoreHead:                  768*256 + 256 + 256*1 + 1     ~= 197 K
    -------------------------------------------------------
    Total trainable           ~= 395 K params

Matches PROJECT_PLAN.md s3.1 "Total trainable ~400K".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import AttentionPooler


@dataclass(frozen=True)
class NatScoreHeadConfig:
    n_hidden_states: int = 13           # 12 transformer layers + 1 embedding for Whisper-small
    hidden_dim: int = 768               # Whisper-small d_model
    pooler_bottleneck_dim: int = 256
    score_bottleneck_dim: int = 256
    dropout: float = 0.0                # zero by default; head is small enough to not overfit fast
    init_layer_weights: str = "uniform"  # "uniform" | "last" | "balanced"


class LayerWeightedSum(nn.Module):
    """Learnable scalar weighting over Whisper encoder layers.

    Parameter count: n_hidden_states (just `alpha`).
    """

    def __init__(self, n_hidden_states: int = 13, init: str = "uniform") -> None:
        super().__init__()
        if init == "uniform":
            init_vals = torch.zeros(n_hidden_states)
        elif init == "last":
            init_vals = torch.full((n_hidden_states,), -5.0)
            init_vals[-1] = 0.0
        elif init == "balanced":
            # bias toward mid-layers, which the layer-wise probe ablation
            # is expected to confirm carry the richest naturalness signal
            mid = n_hidden_states // 2
            init_vals = -((torch.arange(n_hidden_states) - mid).float() ** 2) / 10.0
        else:
            raise ValueError(f"Unknown init={init!r}")
        self.alpha = nn.Parameter(init_vals)

    @property
    def weights(self) -> torch.Tensor:
        """Softmax-normalized weights over the H axis. Shape [H]."""
        return F.softmax(self.alpha, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce [B, H, T, D] -> [B, T, D]."""
        if x.ndim != 4:
            raise ValueError(f"expected [B, H, T, D]; got {tuple(x.shape)}")
        w = self.weights.to(dtype=x.dtype, device=x.device)
        return torch.einsum("h,bhtd->btd", w, x)


class NatScoreHead(nn.Module):
    """Frozen-Whisper-features -> scalar naturalness logit."""

    def __init__(self, config: NatScoreHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or NatScoreHeadConfig()
        c = self.config

        self.layer_sum = LayerWeightedSum(
            n_hidden_states=c.n_hidden_states,
            init=c.init_layer_weights,
        )
        self.pooler = AttentionPooler(
            hidden_dim=c.hidden_dim,
            bottleneck_dim=c.pooler_bottleneck_dim,
        )
        self.dropout = nn.Dropout(c.dropout) if c.dropout > 0 else nn.Identity()
        self.score = nn.Sequential(
            nn.Linear(c.hidden_dim, c.score_bottleneck_dim),
            nn.GELU(),
            nn.Linear(c.score_bottleneck_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, H, T, D] -> [B] scalar logits."""
        x = self.layer_sum(hidden_states)             # [B, T, D]
        pooled = self.pooler(x, valid_frames=valid_frames)  # [B, D]
        pooled = self.dropout(pooled)
        return self.score(pooled).squeeze(-1)         # [B]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
