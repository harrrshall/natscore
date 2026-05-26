"""Bradley-Terry training losses for NatScore.

PROJECT_PLAN.md s3.2:

    Vanilla:        L = -log sigma(s_chosen - s_rejected)
    Magnitude-wt:   L = w * -log sigma(s_chosen - s_rejected)
    Margin:         L = max(0, m - (s_chosen - s_rejected))

The magnitude variant was added after Milestone 1 revealed that
SpeechJudge-Data ships per-rater ordinal annotations like ["B+2", "B+1"]
in the `naturalness_annotation` column. The number after the +/- sign is
a 0-3 magnitude of preference. We average across raters who agree with
the dominant `naturalness_label` to get the per-pair weight w in [0.5, 3].

Why we hand-roll BT instead of using torch.nn.functional.binary_cross_entropy_with_logits:
- it's the same gradient (logsigmoid is more numerically stable than
  log + sigmoid), but encoding `(s_chosen - s_rejected)` directly makes
  the contract crystal clear in code review.
- accumulating w *AS A SCALAR PER PAIR* (not a per-batch reweighting)
  requires us to multiply before reducing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F


_ANNOTATION_RE = re.compile(r"^\s*(?P<side>[AB])\s*\+\s*(?P<mag>\d+)\s*$")


@dataclass(frozen=True)
class BTLossOutput:
    loss: torch.Tensor                # scalar
    accuracy: torch.Tensor            # scalar, fraction of pairs where s_chosen > s_rejected
    mean_margin: torch.Tensor         # scalar, mean(s_chosen - s_rejected)


def bradley_terry_loss(
    s_chosen: torch.Tensor,
    s_rejected: torch.Tensor,
    weight: torch.Tensor | None = None,
    margin: float | None = None,
    reduction: str = "mean",
) -> BTLossOutput:
    """Compute the BT loss + a few free training signals.

    Args:
        s_chosen, s_rejected: scalar logits per pair, shape [B].
        weight: optional per-pair weight shape [B]; None means uniform.
        margin: if provided, add `max(0, margin - delta)` (hinge style).
        reduction: 'mean' | 'sum' | 'none'.
    """
    if s_chosen.shape != s_rejected.shape:
        raise ValueError(
            f"shape mismatch: s_chosen {tuple(s_chosen.shape)} vs s_rejected {tuple(s_rejected.shape)}"
        )
    if s_chosen.ndim != 1:
        raise ValueError(f"expected 1D [B] logits; got shape {tuple(s_chosen.shape)}")

    delta = s_chosen - s_rejected                              # [B]
    per_pair = -F.logsigmoid(delta)                            # [B]
    if margin is not None:
        per_pair = per_pair + F.relu(margin - delta)

    if weight is not None:
        if weight.shape != s_chosen.shape:
            raise ValueError(
                f"weight shape {tuple(weight.shape)} != logits shape {tuple(s_chosen.shape)}"
            )
        per_pair = per_pair * weight

    if reduction == "mean":
        loss = per_pair.mean()
    elif reduction == "sum":
        loss = per_pair.sum()
    elif reduction == "none":
        loss = per_pair
    else:
        raise ValueError(f"reduction must be one of mean|sum|none; got {reduction!r}")

    with torch.no_grad():
        accuracy = (delta > 0).float().mean()
        mean_margin = delta.mean()

    return BTLossOutput(loss=loss, accuracy=accuracy, mean_margin=mean_margin)


def parse_magnitude_weight(
    annotations: list[str],
    naturalness_label: str,
    *,
    floor: float = 0.5,
    cap: float = 3.0,
) -> float:
    """Derive a confidence weight from per-rater ordinal annotations.

    Each annotation has the form "<A|B>+<magnitude>" (e.g. "B+2"). We
    keep only the annotations that agree with the dominant `naturalness_label`
    (the BT label that downstream code uses to pick chosen vs rejected),
    average their magnitudes, then clamp into [floor, cap]. Missing /
    malformed annotations contribute zero weight to the average.

    Returns 1.0 when no usable annotations are found (the safe default).
    """
    if not annotations or not naturalness_label:
        return 1.0
    mags: list[float] = []
    for raw in annotations:
        if not isinstance(raw, str):
            continue
        m = _ANNOTATION_RE.match(raw)
        if not m:
            continue
        if m.group("side") != naturalness_label:
            continue
        mags.append(float(m.group("mag")))
    if not mags:
        return 1.0
    w = sum(mags) / len(mags)
    if w < floor:
        return floor
    if w > cap:
        return cap
    return w
