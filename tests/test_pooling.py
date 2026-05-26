"""Unit tests for the attention pooler."""

from __future__ import annotations

import pytest
import torch

from natscore.pooling import AttentionPooler


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def test_pooler_output_shape():
    _seed()
    pool = AttentionPooler(hidden_dim=768, bottleneck_dim=256)
    x = torch.randn(4, 1500, 768)
    y = pool(x)
    assert y.shape == (4, 768)
    assert torch.isfinite(y).all()


def test_pooler_smaller_dim_works():
    _seed()
    pool = AttentionPooler(hidden_dim=64, bottleneck_dim=16)
    x = torch.randn(2, 50, 64)
    y = pool(x)
    assert y.shape == (2, 64)


def test_pooler_rejects_bad_input_rank():
    pool = AttentionPooler()
    with pytest.raises(ValueError, match="expected"):
        pool(torch.zeros(4, 1500))


def test_pooler_mask_ignores_padded_frames():
    _seed(1)
    pool = AttentionPooler(hidden_dim=8, bottleneck_dim=4)
    x = torch.randn(2, 10, 8)

    # Item 0 valid for first 5 frames; corrupt later frames with huge values
    # the pooler must NOT attend to. Item 1 valid for all 10.
    x_corrupted = x.clone()
    x_corrupted[0, 5:] = 1e6  # would dominate any unmasked attention
    valid = torch.tensor([5, 10])
    y_masked = pool(x_corrupted, valid_frames=valid)
    y_clean = pool(x[:, :5], valid_frames=torch.tensor([5, 5]))
    # The first item should match what we'd get from pooling just the
    # first 5 frames, because the rest are masked out.
    assert torch.allclose(y_masked[0], y_clean[0], atol=1e-5)


def test_pooler_zero_valid_frames_does_not_nan():
    pool = AttentionPooler(hidden_dim=8, bottleneck_dim=4)
    x = torch.randn(1, 10, 8)
    y = pool(x, valid_frames=torch.tensor([0]))
    assert torch.isfinite(y).all(), "pooler produced NaN for an all-masked clip"


def test_pooler_deterministic():
    _seed(7)
    pool = AttentionPooler()
    x = torch.randn(3, 100, 768)
    a = pool(x)
    b = pool(x)
    assert torch.equal(a, b)


def test_pooler_attention_weights_sum_to_one():
    """Direct check of softmax invariant on the bottleneck output."""
    _seed()
    pool = AttentionPooler(hidden_dim=64, bottleneck_dim=16)
    x = torch.randn(3, 50, 64)
    logits = pool.score(x).squeeze(-1)
    attn = torch.softmax(logits, dim=-1)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
