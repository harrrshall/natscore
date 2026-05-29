"""Unit tests for `natscore.compare.Pair`."""

from __future__ import annotations

import math

import pytest

from natscore.compare import Pair


def test_a_wins_when_score_a_higher() -> None:
    p = Pair.from_scores(1.5, -0.5)
    assert p.winner == "a"
    assert p.margin == pytest.approx(2.0)
    assert p.prob_a_wins > 0.5


def test_b_wins_when_score_b_higher() -> None:
    p = Pair.from_scores(-0.5, 1.5)
    assert p.winner == "b"
    assert p.margin == pytest.approx(-2.0)
    assert p.prob_a_wins < 0.5


def test_tie_within_eps() -> None:
    p = Pair.from_scores(1.0, 1.0)
    assert p.winner == "tie"
    assert p.margin == pytest.approx(0.0)
    assert p.prob_a_wins == pytest.approx(0.5)


def test_prob_is_sigmoid_of_margin() -> None:
    p = Pair.from_scores(2.0, 0.5)
    expected = 1.0 / (1.0 + math.exp(-(2.0 - 0.5)))
    assert p.prob_a_wins == pytest.approx(expected, rel=1e-6)


def test_prob_numerically_stable_at_extremes() -> None:
    # Without the stable-sigmoid branch, exp(700) overflows to inf.
    p_large = Pair.from_scores(700.0, -700.0)
    assert 0.0 < p_large.prob_a_wins <= 1.0
    p_small = Pair.from_scores(-700.0, 700.0)
    assert 0.0 <= p_small.prob_a_wins < 1.0


def test_pair_is_frozen() -> None:
    p = Pair.from_scores(1.0, 0.0)
    with pytest.raises(Exception):
        p.score_a = 99.0  # type: ignore[misc]


def test_fields_coerced_to_float() -> None:
    import numpy as np

    p = Pair.from_scores(np.float32(1.5), np.float32(-0.5))
    assert isinstance(p.score_a, float)
    assert isinstance(p.score_b, float)
    assert isinstance(p.margin, float)
    assert isinstance(p.prob_a_wins, float)
