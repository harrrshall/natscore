"""Expected Calibration Error for the BT-confidence sigmoid.

For each pair we compute confidence = sigmoid(|s_chosen - s_rejected|).
A perfectly-calibrated scorer's confidence equals its accuracy at that
confidence level. ECE measures the weighted gap between predicted
confidence and observed accuracy across equal-frequency bins.

Useful for understanding whether a low-pairwise-accuracy run is
"confidently wrong" (bad) vs "uncertain and wrong" (potentially fixable
with anchor regression -- PROJECT_PLAN.md s3.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    ece: float
    n_pairs: int
    n_bins: int
    bin_confidence: list[float]            # mean confidence per bin
    bin_accuracy: list[float]              # mean accuracy per bin
    bin_count: list[int]                   # number of pairs in each bin


def expected_calibration_error(
    margins: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """ECE over absolute-margin sigmoid confidence.

    Args:
        margins: 1-D array of (s_chosen - s_rejected) values per pair.
        correct: 1-D array of 0/1 indicating whether the predicted
                 ordering matched the label (delta > 0).
        n_bins: number of equal-width confidence bins in [0.5, 1.0].

    Returns:
        CalibrationResult with the scalar ECE and per-bin breakdown
        suitable for plotting a reliability diagram.
    """
    if margins.shape != correct.shape:
        raise ValueError(
            f"shape mismatch: margins {margins.shape} vs correct {correct.shape}"
        )
    if margins.ndim != 1:
        raise ValueError(f"expected 1-D arrays; got {margins.ndim}-D")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")

    n = margins.size
    # Confidence in the predicted side is sigmoid of the absolute margin
    # (the model is symmetric: pred ordering is sign(delta), confidence is
    # how decisive |delta| is).
    confidence = 1.0 / (1.0 + np.exp(-np.abs(margins)))
    # `correct` is 1 if the predicted side equals the chosen side, 0 else.
    # For bins we want accuracy at each confidence level.

    # Equal-width bins on [0.5, 1.0]. lower-inclusive, upper-inclusive on the last bin.
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece_total = 0.0
    bin_conf: list[float] = []
    bin_acc: list[float] = []
    bin_cnt: list[int] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence >= lo) & (confidence < hi)
        count = int(in_bin.sum())
        if count == 0:
            bin_conf.append(float("nan"))
            bin_acc.append(float("nan"))
            bin_cnt.append(0)
            continue
        c_mean = float(confidence[in_bin].mean())
        a_mean = float(correct[in_bin].astype(np.float64).mean())
        bin_conf.append(c_mean)
        bin_acc.append(a_mean)
        bin_cnt.append(count)
        ece_total += (count / n) * abs(c_mean - a_mean)

    return CalibrationResult(
        ece=ece_total,
        n_pairs=int(n),
        n_bins=n_bins,
        bin_confidence=bin_conf,
        bin_accuracy=bin_acc,
        bin_count=bin_cnt,
    )
