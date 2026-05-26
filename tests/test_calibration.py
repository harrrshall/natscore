"""Unit tests for expected calibration error."""

from __future__ import annotations

import math

import numpy as np
import pytest

from natscore.eval.calibration import expected_calibration_error


def test_ece_perfect_calibration_is_zero():
    """Confidence == accuracy in every bin -> ECE = 0."""
    # Two clusters of pairs, each perfectly calibrated.
    # Cluster 1: 4 pairs with confidence ~0.6, accuracy 0.5
    #   margins giving sigmoid ~= 0.6  =>  m = ln(0.6/0.4) ~= 0.4055
    m1 = math.log(0.6 / 0.4)
    margins_a = np.full(4, m1)
    # 2 correct, 2 wrong  -> mean acc 0.5 (but bin mean confidence = 0.6)
    # That isn't perfectly calibrated. Build something simpler:

    # Use one bin with confidence 1.0 and accuracy 1.0:
    margins = np.full(10, 100.0)  # confidence ~= 1.0
    correct = np.ones(10)
    out = expected_calibration_error(margins, correct, n_bins=10)
    assert out.ece < 1e-6


def test_ece_max_miscalibration_high():
    """Pairs are 100% confident but always wrong -> ECE ~ 1.0 in that bin."""
    margins = np.full(10, 100.0)         # confidence ~= 1.0
    correct = np.zeros(10)
    out = expected_calibration_error(margins, correct, n_bins=10)
    # Single bin with conf~1, acc=0 -> bin gap = 1
    assert out.ece > 0.95


def test_ece_zero_confidence_bin():
    """All margins ~0 -> confidence ~0.5; ECE = |0.5 - accuracy|."""
    margins = np.zeros(100)
    correct = np.zeros(100)
    out = expected_calibration_error(margins, correct, n_bins=10)
    # confidence ~0.5, accuracy 0 -> bin gap = 0.5
    assert abs(out.ece - 0.5) < 0.05


def test_ece_handles_uneven_bins():
    """Multiple bins with different sample counts -> weighted average."""
    margins = np.concatenate([np.full(80, 100.0), np.full(20, 0.0)])
    correct = np.concatenate([np.ones(80), np.zeros(20)])
    # Bin near 1.0: 80 pairs, conf~1, acc 1, gap 0
    # Bin near 0.5: 20 pairs, conf~0.5, acc 0, gap 0.5
    # ECE = 0.8 * 0 + 0.2 * 0.5 = 0.1
    out = expected_calibration_error(margins, correct, n_bins=10)
    assert abs(out.ece - 0.1) < 0.05


def test_ece_bin_counts_sum_to_n():
    margins = np.linspace(-3.0, 3.0, 50)
    correct = (margins > 0).astype(np.int64)
    out = expected_calibration_error(margins, correct, n_bins=10)
    assert sum(out.bin_count) == 50
    assert out.n_pairs == 50


def test_ece_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_calibration_error(np.zeros(3), np.zeros(4))


def test_ece_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        expected_calibration_error(np.zeros((3, 3)), np.zeros((3, 3)))


def test_ece_rejects_bad_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(np.zeros(5), np.zeros(5), n_bins=0)


def test_ece_returns_correct_dataclass_fields():
    margins = np.linspace(-1.0, 1.0, 20)
    correct = (margins > 0).astype(np.int64)
    out = expected_calibration_error(margins, correct, n_bins=5)
    assert out.n_bins == 5
    assert len(out.bin_confidence) == 5
    assert len(out.bin_accuracy) == 5
    assert len(out.bin_count) == 5
    assert 0.0 <= out.ece <= 1.0
