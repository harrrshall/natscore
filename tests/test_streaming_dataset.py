"""Unit tests for the online streaming pair dataset.

Uses a hand-built iterator of PairRecords -- no HF dataset, no actual
parquet -- to exercise the wrapper logic.
"""

from __future__ import annotations

import io
import wave
from unittest.mock import patch

import numpy as np
import pytest
import torch

from natscore.data.online_pair_dataset import (
    WHISPER_MAX_SAMPLES,
    WHISPER_SR,
    StreamingPairDataset,
    _resample_and_trim,
    collate_streaming_pairs,
)
from natscore.data.speechjudge import PairRecord


def _wav(seconds: float, sr: int = WHISPER_SR, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _pair(idx: int, *, label: str = "B", annotations=None, chosen: bool = True,
          subset: str = "regular", language_setting: str = "en2en",
          sr: int = WHISPER_SR, seconds: float = 1.0) -> PairRecord:
    return PairRecord(
        pair_index=idx, subset=subset, language_setting=language_setting,
        target_text="t", audio_a=_wav(seconds, sr=sr, freq=110.0),
        audio_b=_wav(seconds, sr=sr, freq=220.0),
        sample_rate_a=sr, sample_rate_b=sr,
        naturalness_label=label,
        naturalness_annotation=annotations or ["B+1"],
        chosen=chosen,
    )


# ----------------------------------------------------------- _resample_and_trim


def test_resample_passthrough_at_16k():
    wav = _wav(2.0)
    out = _resample_and_trim(wav, WHISPER_SR)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape == wav.shape


def test_resample_downsamples_from_24k():
    wav_24k = _wav(1.0, sr=24_000)
    out = _resample_and_trim(wav_24k, 24_000)
    expected_len = int(1.0 * WHISPER_SR)
    # librosa rounds; allow a few samples slack
    assert abs(out.shape[0] - expected_len) <= 8


def test_resample_caps_at_30_seconds():
    wav = _wav(45.0)
    out = _resample_and_trim(wav, WHISPER_SR)
    assert out.shape[0] == WHISPER_MAX_SAMPLES


def test_resample_handles_stereo_via_downmix():
    stereo = np.stack([_wav(1.0), _wav(1.0)], axis=1)
    out = _resample_and_trim(stereo, WHISPER_SR)
    assert out.ndim == 1


# ------------------------------------------------------- StreamingPairDataset


def test_streaming_dataset_picks_chosen_from_label():
    ds = StreamingPairDataset(limit=2)
    # Patch iter_pairs to a synthetic stream.
    pairs = [
        _pair(0, label="B"),
        _pair(1, label="A"),
    ]
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        items = list(ds)
    assert len(items) == 2
    # pair 0: label B -> chosen is audio_b (freq 220), rejected is audio_a (freq 110)
    assert items[0].wav_chosen[100] != items[0].wav_rejected[100]
    # pair 1: label A -> chosen is audio_a, rejected is audio_b (flipped)
    assert items[1].wav_chosen[100] != items[1].wav_rejected[100]


def test_streaming_dataset_skips_unknown_label():
    ds = StreamingPairDataset()
    pairs = [_pair(0, label=""), _pair(1, label="B")]
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        items = list(ds)
    assert len(items) == 1
    assert items[0].pair_index == 1


def test_streaming_dataset_high_consensus_filter():
    pairs = [
        _pair(0, label="B", chosen=True),
        _pair(1, label="B", chosen=False),
    ]
    ds_all = StreamingPairDataset(high_consensus_only=False)
    ds_strict = StreamingPairDataset(high_consensus_only=True)
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        all_items = list(ds_all)
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        strict_items = list(ds_strict)
    assert len(all_items) == 2
    assert len(strict_items) == 1


def test_streaming_dataset_magnitude_weighting():
    pairs = [_pair(0, label="B", annotations=["B+2", "B+1"])]
    ds = StreamingPairDataset(magnitude_weighting=True)
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        items = list(ds)
    assert items[0].weight == pytest.approx(1.5)


def test_streaming_dataset_default_weight_is_one():
    pairs = [_pair(0, label="B", annotations=["B+2", "B+1"])]
    ds = StreamingPairDataset(magnitude_weighting=False)
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        items = list(ds)
    assert items[0].weight == 1.0


def test_streaming_dataset_set_epoch_advances_shard_seed():
    ds = StreamingPairDataset(shuffle_shards=True, shard_seed=42)
    assert ds._epoch == 0
    ds.set_epoch(3)
    assert ds._epoch == 3


# --------------------------------------------------------------- collate


def test_collate_streaming_pairs_packs_correctly():
    pairs = [_pair(0, label="B"), _pair(1, label="A", seconds=0.5)]
    ds = StreamingPairDataset()
    with patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(pairs)):
        items = list(ds)
    batch = collate_streaming_pairs(items)
    assert batch["pair_index"].tolist() == [0, 1]
    assert len(batch["wav_chosen"]) == 2
    assert len(batch["wav_rejected"]) == 2
    assert batch["weight"].shape == (2,)
    assert isinstance(batch["wav_chosen"][0], np.ndarray)
    # Items have different lengths -- this is intentional; HF mel pipeline
    # will pad to 30s. So shapes must differ here.
    assert batch["wav_chosen"][0].shape != batch["wav_chosen"][1].shape
