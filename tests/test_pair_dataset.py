"""Unit tests for the PairDataset / collate pipeline.

Synthetic features only -- no Whisper, no dataset, no audio.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from natscore.data.feature_cache import FeatureCache
from natscore.data.pair_dataset import PairDataset, PairMeta, collate_pairs
from natscore.train.losses import parse_magnitude_weight


def _write_pair(cache: FeatureCache, idx: int, label_a: float, label_b: float) -> None:
    """Stash two synthetic feature tensors; the value encodes A/B so tests can detect mix-ups."""
    feat_a = torch.full((3, 5, 4), label_a, dtype=torch.float16)
    feat_b = torch.full((3, 5, 4), label_b, dtype=torch.float16)
    cache.write(
        clip_id=f"{idx:08d}_A", hidden_states=feat_a,
        pair_index=idx, side="A", subset="r", language_setting="en2en",
        sample_rate=24_000, duration_seconds=2.0, target_text="ta",
    )
    cache.write(
        clip_id=f"{idx:08d}_B", hidden_states=feat_b,
        pair_index=idx, side="B", subset="r", language_setting="en2en",
        sample_rate=24_000, duration_seconds=2.0, target_text="tb",
    )


def test_pair_dataset_joins_a_and_b_correctly(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair(cache, 0, 1.0, 2.0)   # A=1, B=2, label B -> chosen is the 2.0 tensor
    _write_pair(cache, 1, 5.0, 3.0)   # A=5, B=3, label A -> chosen is the 5.0 tensor
    meta = {
        0: PairMeta(0, "B", ["B+1"], True),
        1: PairMeta(1, "A", ["A+2"], True),
    }
    ds = PairDataset(tmp_path, pair_meta=meta)
    assert len(ds) == 2

    p0 = ds[0]
    assert torch.allclose(p0.feat_chosen.mean(), torch.tensor(2.0))
    assert torch.allclose(p0.feat_rejected.mean(), torch.tensor(1.0))

    p1 = ds[1]
    assert torch.allclose(p1.feat_chosen.mean(), torch.tensor(5.0))
    assert torch.allclose(p1.feat_rejected.mean(), torch.tensor(3.0))


def test_pair_dataset_high_consensus_filter(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair(cache, 0, 1, 2)
    _write_pair(cache, 1, 3, 4)
    meta = {
        0: PairMeta(0, "A", ["A+1"], True),
        1: PairMeta(1, "B", ["B+1"], False),  # low consensus -> dropped
    }
    ds_all = PairDataset(tmp_path, pair_meta=meta, high_consensus_only=False)
    ds_strict = PairDataset(tmp_path, pair_meta=meta, high_consensus_only=True)
    assert len(ds_all) == 2
    assert len(ds_strict) == 1


def test_pair_dataset_magnitude_weighting(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair(cache, 0, 1, 2)
    meta = {0: PairMeta(0, "B", ["B+2", "B+1"], True)}
    ds_unw = PairDataset(tmp_path, pair_meta=meta, magnitude_weighting=False)
    ds_w = PairDataset(tmp_path, pair_meta=meta, magnitude_weighting=True)
    assert ds_unw[0].weight == 1.0
    assert ds_w[0].weight == pytest.approx(1.5)


def test_pair_dataset_skips_pairs_with_unknown_label(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair(cache, 0, 1, 2)
    meta = {0: PairMeta(0, "", [], False)}  # unusable label
    ds = PairDataset(tmp_path, pair_meta=meta)
    assert len(ds) == 0


def test_pair_dataset_drops_pairs_missing_a_side(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    # write only side A for pair 0
    cache.write(
        clip_id="00000000_A", hidden_states=torch.zeros(3, 5, 4, dtype=torch.float16),
        pair_index=0, side="A", subset="r", language_setting="en2en",
        sample_rate=24_000, duration_seconds=1.0, target_text="",
    )
    meta = {0: PairMeta(0, "A", ["A+1"], True)}
    ds = PairDataset(tmp_path, pair_meta=meta, drop_pairs_missing_side=True)
    assert len(ds) == 0
    with pytest.raises(ValueError, match="missing side"):
        PairDataset(tmp_path, pair_meta=meta, drop_pairs_missing_side=False)


def test_collate_pairs_stacks_correctly(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    _write_pair(cache, 0, 1, 2)
    _write_pair(cache, 1, 3, 4)
    meta = {
        0: PairMeta(0, "A", ["A+1"], True),
        1: PairMeta(1, "B", ["B+2"], True),
    }
    ds = PairDataset(tmp_path, pair_meta=meta)
    batch = collate_pairs([ds[0], ds[1]])
    assert batch["feat_chosen"].shape == (2, 3, 5, 4)
    assert batch["feat_rejected"].shape == (2, 3, 5, 4)
    assert batch["weight"].shape == (2,)
    assert batch["pair_index"].tolist() == [0, 1]
