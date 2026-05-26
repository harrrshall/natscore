"""Unit tests for NatScoreHead and the layer-weighted sum module."""

from __future__ import annotations

import pytest
import torch

from natscore.model import LayerWeightedSum, NatScoreHead, NatScoreHeadConfig


# --------------------------------------------------------------- LayerWeightedSum


def test_layer_weighted_sum_shape_and_softmax():
    lws = LayerWeightedSum(n_hidden_states=13, init="uniform")
    x = torch.randn(2, 13, 1500, 768)
    y = lws(x)
    assert y.shape == (2, 1500, 768)
    w = lws.weights
    assert w.shape == (13,)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
    assert (w >= 0).all()


def test_layer_weighted_sum_uniform_init_equals_mean():
    lws = LayerWeightedSum(n_hidden_states=13, init="uniform")
    x = torch.randn(2, 13, 100, 8)
    y = lws(x)
    expected = x.mean(dim=1)
    assert torch.allclose(y, expected, atol=1e-6), "uniform init should reduce to per-layer mean"


def test_layer_weighted_sum_last_init_peaks_on_last_layer():
    lws = LayerWeightedSum(n_hidden_states=13, init="last")
    w = lws.weights
    assert w.argmax().item() == 12
    assert w[-1] > 0.9, "init=last should put nearly all mass on the last layer"


def test_layer_weighted_sum_balanced_init_peaks_in_middle():
    lws = LayerWeightedSum(n_hidden_states=13, init="balanced")
    w = lws.weights
    assert 5 <= w.argmax().item() <= 7


def test_layer_weighted_sum_invalid_init_raises():
    with pytest.raises(ValueError, match="init"):
        LayerWeightedSum(n_hidden_states=13, init="nope")


def test_layer_weighted_sum_rejects_wrong_rank():
    lws = LayerWeightedSum(n_hidden_states=13)
    with pytest.raises(ValueError, match="expected"):
        lws(torch.zeros(2, 1500, 768))


# ------------------------------------------------------------------ NatScoreHead


def test_natscore_head_forward_shape():
    head = NatScoreHead()
    x = torch.randn(3, 13, 1500, 768)
    s = head(x)
    assert s.shape == (3,)
    assert torch.isfinite(s).all()


def test_natscore_head_param_budget_near_plan():
    head = NatScoreHead()
    n = head.trainable_param_count()
    # PROJECT_PLAN.md s3.1 says "~400K trainable". Be generous: 300K-700K.
    assert 300_000 < n < 700_000, f"head has {n} trainable params; expected ~400K"


def test_natscore_head_deterministic_given_seed():
    torch.manual_seed(0)
    head_a = NatScoreHead()
    torch.manual_seed(0)
    head_b = NatScoreHead()
    x = torch.randn(2, 13, 50, 768)
    sa = head_a(x)
    sb = head_b(x)
    assert torch.equal(sa, sb)


def test_natscore_head_gradient_flows_to_alpha():
    head = NatScoreHead()
    x = torch.randn(2, 13, 100, 768)
    s = head(x).sum()
    s.backward()
    grad = head.layer_sum.alpha.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "alpha got zero gradient"


def test_natscore_head_respects_valid_frames():
    torch.manual_seed(0)
    head = NatScoreHead()
    x = torch.randn(2, 13, 200, 768)
    # Confirm result depends on the mask: same input, different mask -> different score.
    s1 = head(x, valid_frames=torch.tensor([200, 200]))
    s2 = head(x, valid_frames=torch.tensor([20, 200]))
    assert not torch.equal(s1, s2)


def test_natscore_head_config_overrides():
    cfg = NatScoreHeadConfig(
        n_hidden_states=5, hidden_dim=64,
        pooler_bottleneck_dim=16, score_bottleneck_dim=16, dropout=0.1,
    )
    head = NatScoreHead(cfg)
    x = torch.randn(4, 5, 100, 64)
    s = head(x)
    assert s.shape == (4,)
    assert torch.isfinite(s).all()
