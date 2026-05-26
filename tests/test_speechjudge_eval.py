"""Unit tests for the SpeechJudge eval module.

Synthetic features only -- builds a fake cache with feature tensors
encoded to a known ordering, then verifies the eval module reports
correct accuracy / margins / breakdowns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from natscore.data.feature_cache import FeatureCache
from natscore.data.pair_dataset import PairMeta
from natscore.eval.speechjudge_eval import _bootstrap_ci, evaluate


class _LastFrameMean(nn.Module):
    """Toy model: score = mean of the last layer, last frame, all dims.

    Lets us craft feature tensors with known scores so test assertions are
    crisp. Implements the same forward signature as NatScoreHead.
    """

    def forward(self, hidden_states: torch.Tensor, valid_frames=None) -> torch.Tensor:
        # hidden_states is [B, H, T, D]; score is mean of the last layer / last frame.
        return hidden_states[:, -1, -1, :].mean(dim=-1)


def _write_pair_with_scores(
    cache: FeatureCache, idx: int, score_a: float, score_b: float,
    *, subset: str = "regular", language_setting: str = "en2en",
) -> None:
    """Put the desired score into the last-layer, last-frame, all-dims slot."""
    feat_a = torch.zeros(3, 5, 4, dtype=torch.float16)
    feat_b = torch.zeros(3, 5, 4, dtype=torch.float16)
    feat_a[-1, -1, :] = score_a
    feat_b[-1, -1, :] = score_b
    cache.write(clip_id=f"{idx:08d}_A", hidden_states=feat_a,
                pair_index=idx, side="A", subset=subset, language_setting=language_setting,
                sample_rate=24000, duration_seconds=2.0, target_text="")
    cache.write(clip_id=f"{idx:08d}_B", hidden_states=feat_b,
                pair_index=idx, side="B", subset=subset, language_setting=language_setting,
                sample_rate=24000, duration_seconds=2.0, target_text="")


def _write_pair_meta(cache_dir: Path, meta: dict[int, PairMeta]) -> None:
    from dataclasses import asdict
    serializable = {str(k): asdict(v) for k, v in meta.items()}
    (cache_dir / "pair_meta.json").write_text(json.dumps(serializable))


# ---------------------------------------------------------------- evaluate()


def test_evaluate_perfect_model(tmp_path: Path):
    """Construct a setup where the toy model is always right -> acc = 1.0."""
    cache = FeatureCache(tmp_path)
    # 5 pairs, label B always, B has higher score -> model picks chosen correctly
    for i in range(5):
        _write_pair_with_scores(cache, i, score_a=0.1, score_b=0.9)
    _write_pair_meta(tmp_path, {
        i: PairMeta(i, "B", ["B+1"], True) for i in range(5)
    })

    model = _LastFrameMean()
    result = evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"))
    assert result.n_pairs == 5
    assert result.pairwise_accuracy == 1.0
    assert result.mean_margin > 0
    assert 0.0 <= result.ci_low <= result.ci_high <= 1.0


def test_evaluate_adversarial_model(tmp_path: Path):
    """Setup where the model picks the WRONG side -> acc = 0.0."""
    cache = FeatureCache(tmp_path)
    # label B always, but A has higher score -> model picks A every time -> wrong
    for i in range(5):
        _write_pair_with_scores(cache, i, score_a=0.9, score_b=0.1)
    _write_pair_meta(tmp_path, {
        i: PairMeta(i, "B", ["B+1"], True) for i in range(5)
    })
    model = _LastFrameMean()
    result = evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"))
    assert result.pairwise_accuracy == 0.0
    assert result.mean_margin < 0


def test_evaluate_mixed_accuracy(tmp_path: Path):
    """Half pairs the model gets right, half wrong -> acc ~ 0.5."""
    cache = FeatureCache(tmp_path)
    # Pairs 0-3: label B, B has higher score (correct prediction)
    for i in range(4):
        _write_pair_with_scores(cache, i, score_a=0.1, score_b=0.9)
    # Pairs 4-7: label B, A has higher score (wrong prediction)
    for i in range(4, 8):
        _write_pair_with_scores(cache, i, score_a=0.9, score_b=0.1)
    _write_pair_meta(tmp_path, {
        i: PairMeta(i, "B", ["B+1"], True) for i in range(8)
    })
    model = _LastFrameMean()
    result = evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"))
    assert result.pairwise_accuracy == 0.5
    assert result.n_pairs == 8


def test_evaluate_per_subset_breakdown(tmp_path: Path):
    """Verify per-subset breakdown counts/accuracy."""
    cache = FeatureCache(tmp_path)
    # 4 regular pairs, all correct
    for i in range(4):
        _write_pair_with_scores(cache, i, score_a=0.1, score_b=0.9, subset="regular")
    # 2 expressive pairs, all wrong
    for i in range(4, 6):
        _write_pair_with_scores(cache, i, score_a=0.9, score_b=0.1, subset="expressive")
    _write_pair_meta(tmp_path, {
        i: PairMeta(i, "B", ["B+1"], True) for i in range(6)
    })
    model = _LastFrameMean()
    result = evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"))
    assert result.per_subset["regular"]["accuracy"] == 1.0
    assert result.per_subset["regular"]["n_pairs"] == 4
    assert result.per_subset["expressive"]["accuracy"] == 0.0
    assert result.per_subset["expressive"]["n_pairs"] == 2


def test_evaluate_per_language_breakdown(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair_with_scores(cache, 0, 0.1, 0.9, language_setting="zh2en")
    _write_pair_with_scores(cache, 1, 0.1, 0.9, language_setting="en2en")
    _write_pair_meta(tmp_path, {
        0: PairMeta(0, "B", ["B+1"], True),
        1: PairMeta(1, "B", ["B+1"], True),
    })
    model = _LastFrameMean()
    result = evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"))
    assert "zh2en" in result.per_language
    assert "en2en" in result.per_language


def test_evaluate_high_consensus_filter(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair_with_scores(cache, 0, 0.1, 0.9)
    _write_pair_with_scores(cache, 1, 0.1, 0.9)
    _write_pair_meta(tmp_path, {
        0: PairMeta(0, "B", ["B+1"], True),    # included
        1: PairMeta(1, "B", ["B+1"], False),   # excluded under high_consensus_only
    })
    model = _LastFrameMean()
    full = evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"))
    strict = evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"),
                      high_consensus_only=True)
    assert full.n_pairs == 2
    assert strict.n_pairs == 1


def test_evaluate_raises_when_no_meta(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair_with_scores(cache, 0, 0.1, 0.9)
    # No pair_meta.json written!
    model = _LastFrameMean()
    with pytest.raises(FileNotFoundError, match="pair_meta"):
        evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"))


def test_evaluate_raises_when_empty_dataset(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair_with_scores(cache, 0, 0.1, 0.9)
    _write_pair_meta(tmp_path, {
        0: PairMeta(0, "", [], False),       # unusable label
    })
    model = _LastFrameMean()
    with pytest.raises(RuntimeError, match="No pairs"):
        evaluate(model, tmp_path, batch_size=2, device=torch.device("cpu"))


# -------------------------------------------------------------------- bootstrap


def test_bootstrap_ci_brackets_mean():
    import numpy as np

    correct = np.array([1.0] * 8 + [0.0] * 2)  # acc = 0.8
    lo, hi = _bootstrap_ci(correct, n_boot=2000, seed=0)
    assert lo <= 0.8 <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_handles_empty():
    import numpy as np
    import math

    lo, hi = _bootstrap_ci(np.array([]), n_boot=100, seed=0)
    assert math.isnan(lo) and math.isnan(hi)
