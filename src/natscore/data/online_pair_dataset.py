"""IterableDataset that streams SpeechJudge pairs and pads waveforms.

For the Kaggle T4 training path (PROJECT_PLAN.md s6.1.1): no feature cache
on disk. The dataset yields raw audio + metadata; the training step runs
the frozen Whisper encoder on GPU.

The collate function resamples-and-pads to a uniform length (the Whisper
mel pipeline will pad to 30s internally anyway, but we want consistent
shapes inside our DataLoader so the GPU batch dim is fixed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset

from natscore.data.speechjudge import PairRecord, iter_pairs
from natscore.train.losses import parse_magnitude_weight

# Match the Whisper feature extractor expectations.
WHISPER_SR = 16_000
WHISPER_MAX_SEC = 30
WHISPER_MAX_SAMPLES = WHISPER_SR * WHISPER_MAX_SEC


@dataclass(frozen=True)
class StreamingPairItem:
    pair_index: int
    wav_chosen: np.ndarray                # float32, 1-D, 16 kHz mono, len <= WHISPER_MAX_SAMPLES
    wav_rejected: np.ndarray
    weight: float
    subset: str
    language_setting: str


class StreamingPairDataset(IterableDataset[StreamingPairItem]):
    """Wraps iter_pairs(); chooses chosen/rejected via naturalness_label.

    Args:
        split: SpeechJudge split.
        limit: cap on yielded pairs (handy for smoke runs).
        token: HF token.
        high_consensus_only: keep only pairs with chosen == True.
        magnitude_weighting: parse `naturalness_annotation` -> per-pair w.
        shuffle_shards: shard-level epoch shuffle.
        shard_seed: base RNG seed for shard order.
    """

    def __init__(
        self,
        *,
        split: str = "train",
        limit: int | None = None,
        token: str | None = None,
        skip: int = 0,
        high_consensus_only: bool = False,
        magnitude_weighting: bool = False,
        shuffle_shards: bool = False,
        shard_seed: int = 0,
    ) -> None:
        super().__init__()
        self.split = split
        self.limit = limit
        self.token = token
        self.skip = skip
        self.high_consensus_only = high_consensus_only
        self.magnitude_weighting = magnitude_weighting
        self.shuffle_shards = shuffle_shards
        self.shard_seed = shard_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the shard-order RNG between epochs."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[StreamingPairItem]:
        for pr in iter_pairs(
            split=self.split, limit=self.limit, token=self.token, skip=self.skip,
            shuffle_shards=self.shuffle_shards,
            shard_seed=self.shard_seed + self._epoch,
        ):
            item = self._to_item(pr)
            if item is not None:
                yield item

    def _to_item(self, pr: PairRecord) -> StreamingPairItem | None:
        if self.high_consensus_only and not pr.chosen:
            return None
        if pr.naturalness_label == "A":
            wav_c, sr_c = pr.audio_a, pr.sample_rate_a
            wav_r, sr_r = pr.audio_b, pr.sample_rate_b
        elif pr.naturalness_label == "B":
            wav_c, sr_c = pr.audio_b, pr.sample_rate_b
            wav_r, sr_r = pr.audio_a, pr.sample_rate_a
        else:
            return None

        wav_c = _resample_and_trim(wav_c, sr_c)
        wav_r = _resample_and_trim(wav_r, sr_r)

        if self.magnitude_weighting:
            w = parse_magnitude_weight(pr.naturalness_annotation, pr.naturalness_label)
        else:
            w = 1.0
        return StreamingPairItem(
            pair_index=pr.pair_index,
            wav_chosen=wav_c, wav_rejected=wav_r,
            weight=w, subset=pr.subset, language_setting=pr.language_setting,
        )


def _resample_and_trim(wav: np.ndarray, sr: int) -> np.ndarray:
    """Force float32 mono at 16 kHz, capped at 30 seconds."""
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)
    if sr != WHISPER_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=WHISPER_SR).astype(np.float32, copy=False)
    if wav.shape[0] > WHISPER_MAX_SAMPLES:
        wav = wav[:WHISPER_MAX_SAMPLES]
    return wav


def collate_streaming_pairs(batch: list[StreamingPairItem]) -> dict:
    """Pack a list of StreamingPairItem into ragged-list dict.

    We pass python lists of numpy arrays straight through to the trainer;
    the WhisperFeatureExtractor's HF feature extractor handles padding to
    30s internally. Keeping arrays ragged avoids double-padding work.
    """
    return {
        "pair_index": torch.tensor([b.pair_index for b in batch], dtype=torch.long),
        "wav_chosen": [b.wav_chosen for b in batch],
        "wav_rejected": [b.wav_rejected for b in batch],
        "weight": torch.tensor([b.weight for b in batch], dtype=torch.float32),
        "subset": [b.subset for b in batch],
        "language_setting": [b.language_setting for b in batch],
    }
