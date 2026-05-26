"""Unit tests for the M5b Tier-1 ablation infrastructure.

Covers:
  - subset_filter on StreamingPairDataset
  - frozen_layer_idx behaviour on LayerWeightedSum / NatScoreHead
  - the four shipped Tier-1 config YAMLs parse + round-trip cleanly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from natscore.data.online_pair_dataset import StreamingPairDataset
from natscore.data.speechjudge import PairRecord
from natscore.model import (
    LayerWeightedSum,
    NatScoreHead,
    NatScoreHeadConfig,
)
from natscore.train.config import TrainConfig

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _pair(idx: int, *, subset: str, label: str = "B") -> PairRecord:
    wav = np.zeros(16000, dtype=np.float32)
    return PairRecord(
        pair_index=idx, subset=subset, language_setting="en2en",
        target_text="t", audio_a=wav, audio_b=wav,
        sample_rate_a=16000, sample_rate_b=16000,
        naturalness_label=label, naturalness_annotation=["B+1"], chosen=True,
    )


# ------------------------------------------------------ subset filter


def test_subset_filter_regular_only():
    pairs = [
        _pair(0, subset="regular"),
        _pair(1, subset="expressive"),
        _pair(2, subset="regular"),
        _pair(3, subset="expressive"),
    ]
    ds = StreamingPairDataset(subset_filter="regular")
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        items = list(ds)
    assert len(items) == 2
    assert {it.pair_index for it in items} == {0, 2}
    assert all(it.subset == "regular" for it in items)


def test_subset_filter_expressive_only():
    pairs = [
        _pair(0, subset="regular"),
        _pair(1, subset="expressive"),
    ]
    ds = StreamingPairDataset(subset_filter="expressive")
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        items = list(ds)
    assert len(items) == 1
    assert items[0].pair_index == 1


def test_subset_filter_none_keeps_all():
    pairs = [
        _pair(0, subset="regular"),
        _pair(1, subset="expressive"),
    ]
    ds = StreamingPairDataset(subset_filter=None)
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        items = list(ds)
    assert len(items) == 2


def test_subset_filter_invalid_raises():
    with pytest.raises(ValueError, match="subset_filter"):
        StreamingPairDataset(subset_filter="bogus")


# ------------------------------------------------------ layer probe freeze


def test_freeze_to_layer_makes_alpha_one_hot():
    lws = LayerWeightedSum(n_hidden_states=13, init="uniform")
    lws.freeze_to_layer(6)
    w = lws.weights
    assert w[6].item() > 0.999
    # other layers near zero
    assert all(w[i].item() < 1e-3 for i in range(13) if i != 6)
    assert lws.alpha.requires_grad is False
    assert lws.frozen_layer_idx == 6


def test_freeze_to_layer_constructor_arg():
    lws = LayerWeightedSum(n_hidden_states=13, init="uniform", frozen_layer_idx=3)
    assert lws.frozen_layer_idx == 3
    assert lws.alpha.requires_grad is False
    assert lws.weights.argmax().item() == 3


def test_freeze_to_layer_rejects_out_of_range():
    lws = LayerWeightedSum(n_hidden_states=13)
    with pytest.raises(ValueError, match="layer_idx="):
        lws.freeze_to_layer(13)
    with pytest.raises(ValueError, match="layer_idx="):
        lws.freeze_to_layer(-1)


def test_natscore_head_with_frozen_layer_reduces_trainable_params():
    cfg_free = NatScoreHeadConfig()
    cfg_frozen = NatScoreHeadConfig(frozen_layer_idx=6)
    h_free = NatScoreHead(cfg_free)
    h_frozen = NatScoreHead(cfg_frozen)
    # Difference == 13 (the alpha vector).
    assert h_free.trainable_param_count() - h_frozen.trainable_param_count() == 13


def test_natscore_head_with_frozen_layer_forward_works():
    cfg = NatScoreHeadConfig(frozen_layer_idx=2)
    head = NatScoreHead(cfg)
    x = torch.randn(2, 13, 100, 768)
    s = head(x)
    assert s.shape == (2,)
    assert torch.isfinite(s).all()


def test_natscore_head_frozen_layer_alpha_gets_no_grad():
    cfg = NatScoreHeadConfig(frozen_layer_idx=0)
    head = NatScoreHead(cfg)
    x = torch.randn(2, 13, 50, 768)
    s = head(x).sum()
    s.backward()
    assert head.layer_sum.alpha.grad is None or head.layer_sum.alpha.grad.abs().sum() == 0


# ------------------------------------------------------ config YAMLs round-trip


@pytest.mark.parametrize("name", [
    "train.kaggle.yaml",
    "train.high_consensus.yaml",
    "train.magnitude.yaml",
    "train.regular_only.yaml",
    "train.layer_probe_L6.yaml",
])
def test_tier1_config_yamls_load_cleanly(name: str):
    cfg = TrainConfig.from_yaml(CONFIG_DIR / name)
    # Each ablation config is identifiable by its unique flag.
    if "high_consensus" in name:
        assert cfg.data.high_consensus_only is True
    elif "magnitude" in name:
        assert cfg.data.magnitude_weighting is True
    elif "regular_only" in name:
        assert cfg.data.subset_filter == "regular"
    elif "layer_probe" in name:
        assert cfg.model.frozen_layer_idx is not None


def test_kaggle_yaml_is_strict_baseline():
    """Sanity: train.kaggle.yaml should have ALL ablation flags off so the
    other configs only diverge by their named feature."""
    cfg = TrainConfig.from_yaml(CONFIG_DIR / "train.kaggle.yaml")
    assert cfg.data.high_consensus_only is False
    assert cfg.data.magnitude_weighting is False
    assert cfg.data.subset_filter is None
    assert cfg.model.frozen_layer_idx is None
