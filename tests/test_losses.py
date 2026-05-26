"""Unit tests for the Bradley-Terry losses and annotation parsing."""

from __future__ import annotations

import math

import pytest
import torch

from natscore.train.losses import (
    BTLossOutput,
    bradley_terry_loss,
    parse_magnitude_weight,
)


def test_bt_loss_returns_expected_dataclass_fields():
    s_c = torch.tensor([1.0, 2.0])
    s_r = torch.tensor([0.0, 1.0])
    out = bradley_terry_loss(s_c, s_r)
    assert isinstance(out, BTLossOutput)
    assert out.loss.ndim == 0
    assert out.accuracy.ndim == 0
    assert out.mean_margin.ndim == 0


def test_bt_loss_zero_when_chosen_dominates_strongly():
    s_c = torch.tensor([100.0])
    s_r = torch.tensor([0.0])
    out = bradley_terry_loss(s_c, s_r)
    assert out.loss.item() < 1e-6
    assert out.accuracy.item() == pytest.approx(1.0)


def test_bt_loss_log2_at_zero_margin():
    s_c = torch.tensor([0.0, 0.0])
    s_r = torch.tensor([0.0, 0.0])
    out = bradley_terry_loss(s_c, s_r)
    # -log(sigmoid(0)) = log(2)
    assert out.loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert out.mean_margin.item() == pytest.approx(0.0)


def test_bt_loss_negative_when_rejected_dominates():
    s_c = torch.tensor([-5.0])
    s_r = torch.tensor([5.0])
    out = bradley_terry_loss(s_c, s_r)
    assert out.loss.item() > 5.0
    assert out.accuracy.item() == 0.0


def test_bt_loss_weight_scales_loss():
    s_c = torch.tensor([0.0, 0.0])
    s_r = torch.tensor([0.0, 0.0])
    w = torch.tensor([1.0, 3.0])
    out = bradley_terry_loss(s_c, s_r, weight=w)
    expected_loss = (1 * math.log(2) + 3 * math.log(2)) / 2
    assert out.loss.item() == pytest.approx(expected_loss, abs=1e-6)


def test_bt_loss_margin_adds_hinge_penalty():
    s_c = torch.tensor([0.5])
    s_r = torch.tensor([0.0])
    out = bradley_terry_loss(s_c, s_r, margin=1.0)
    out_no_margin = bradley_terry_loss(s_c, s_r)
    assert out.loss.item() > out_no_margin.loss.item()


def test_bt_loss_reductions_match():
    s_c = torch.tensor([1.0, 0.5, -0.5])
    s_r = torch.tensor([0.0, 0.5, 0.0])
    none_out = bradley_terry_loss(s_c, s_r, reduction="none")
    sum_out = bradley_terry_loss(s_c, s_r, reduction="sum")
    mean_out = bradley_terry_loss(s_c, s_r, reduction="mean")
    assert none_out.loss.shape == (3,)
    assert sum_out.loss.item() == pytest.approx(none_out.loss.sum().item(), abs=1e-6)
    assert mean_out.loss.item() == pytest.approx(none_out.loss.mean().item(), abs=1e-6)


def test_bt_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        bradley_terry_loss(torch.zeros(3), torch.zeros(4))


def test_bt_loss_rejects_rank_mismatch():
    with pytest.raises(ValueError, match="1D"):
        bradley_terry_loss(torch.zeros(3, 3), torch.zeros(3, 3))


# --------------------------------------------------- parse_magnitude_weight


def test_parse_magnitude_weight_simple_agreement():
    w = parse_magnitude_weight(["B+2", "B+1"], "B")
    assert w == pytest.approx(1.5)


def test_parse_magnitude_weight_ignores_disagreeing_raters():
    w = parse_magnitude_weight(["B+2", "A+1"], "B")
    assert w == pytest.approx(2.0)


def test_parse_magnitude_weight_defaults_when_empty():
    assert parse_magnitude_weight([], "B") == 1.0
    assert parse_magnitude_weight(["A+1"], "B") == 1.0
    assert parse_magnitude_weight(["bogus"], "B") == 1.0
    assert parse_magnitude_weight(None, "B") == 1.0  # type: ignore[arg-type]


def test_parse_magnitude_weight_clamps_range():
    w = parse_magnitude_weight(["B+0"], "B")
    assert w == 0.5  # floor
    w = parse_magnitude_weight(["B+99"], "B")
    assert w == 3.0  # cap


def test_parse_magnitude_weight_handles_whitespace_and_case():
    w = parse_magnitude_weight(["  B + 1 ", "B+2"], "B")
    assert w == pytest.approx(1.5)
