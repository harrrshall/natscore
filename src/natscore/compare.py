"""Pair: structured result of `Scorer.compare(a, b)`.

A frozen dataclass so callers can pattern-match or destructure without
worrying about the underlying scorer's internal representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


Winner = Literal["a", "b", "tie"]


@dataclass(frozen=True)
class Pair:
    """Result of comparing two audio clips with a NatScore Scorer.

    Attributes:
        score_a: raw naturalness logit for clip A (higher = more natural).
        score_b: raw naturalness logit for clip B.
        margin: score_a - score_b. Sign + magnitude = preference + confidence.
        winner: "a" if a > b, "b" if b > a, "tie" if equal within float eps.
        prob_a_wins: Bradley-Terry probability that A is preferred,
            sigmoid(score_a - score_b). Lives in (0, 1).
    """

    score_a: float
    score_b: float
    margin: float
    winner: Winner
    prob_a_wins: float

    @classmethod
    def from_scores(cls, score_a: float, score_b: float, eps: float = 1e-9) -> "Pair":
        margin = score_a - score_b
        if abs(margin) < eps:
            winner: Winner = "tie"
        elif margin > 0:
            winner = "a"
        else:
            winner = "b"
        # Numerically-stable sigmoid (avoids overflow at large |margin|)
        if margin >= 0:
            prob = 1.0 / (1.0 + math.exp(-margin))
        else:
            ex = math.exp(margin)
            prob = ex / (1.0 + ex)
        return cls(
            score_a=float(score_a),
            score_b=float(score_b),
            margin=float(margin),
            winner=winner,
            prob_a_wins=prob,
        )
