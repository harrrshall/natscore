"""Smoke tests for OnlineTrainer with the Whisper encoder mocked.

Verifies the wiring (encoder forward -> head forward -> BT loss -> optimizer
step -> checkpoint round-trip) without paying the ~150 MB Whisper download
or a real GPU. The real end-to-end validation happens on Kaggle T4.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from natscore.data.online_pair_dataset import StreamingPairItem
from natscore.data.speechjudge import PairRecord
from natscore.features import WhisperFeatureMeta
from natscore.train.config import TrainConfig


class _FakeExtractor:
    """Stands in for WhisperFeatureExtractor: tiny deterministic features."""

    def __init__(self, *args, **kwargs) -> None:
        self.meta = WhisperFeatureMeta(
            model_name="fake", n_layers=2, n_hidden_states=3,
            hidden_dim=8, sample_rate=16_000, frame_rate=50,
            max_audio_seconds=30, output_frames=10,
        )

    def batch_extract_layerwise(self, audios: list) -> torch.Tensor:
        # Deterministic: each clip becomes a tensor whose mean reflects its
        # first sample. Lets us craft pairs the trainer must learn to order.
        feats = []
        for a in audios:
            seed = float(a[0]) if len(a) else 0.0
            base = torch.full((3, 10, 8), seed, dtype=torch.float32)
            feats.append(base)
        return torch.stack(feats, dim=0).to("cpu")


def _wav(value: float, samples: int = 100) -> np.ndarray:
    """Produce a waveform whose first sample equals `value`."""
    w = np.full(samples, value, dtype=np.float32)
    return w


def _pair(idx: int, score_for_chosen: float, score_for_rejected: float, *,
          label: str = "B") -> PairRecord:
    """When label='B', audio_b is chosen, audio_a is rejected."""
    if label == "B":
        wav_a = _wav(score_for_rejected); wav_b = _wav(score_for_chosen)
    else:
        wav_a = _wav(score_for_chosen);   wav_b = _wav(score_for_rejected)
    return PairRecord(
        pair_index=idx, subset="regular", language_setting="en2en",
        target_text="t", audio_a=wav_a, audio_b=wav_b,
        sample_rate_a=16_000, sample_rate_b=16_000,
        naturalness_label=label, naturalness_annotation=["B+1"], chosen=True,
    )


def test_online_trainer_constructs_and_takes_one_step(tmp_path: Path):
    cfg = TrainConfig.from_dict({
        "run_name": "online-smoke", "seed": 0, "batch_size": 2,
        "epochs": 1, "max_steps": 5, "log_every_steps": 1,
        "checkpoint_every_steps": 0,
        "output_dir": str(tmp_path), "wandb_enabled": False,
        "model": {"n_hidden_states": 3, "hidden_dim": 8,
                  "pooler_bottleneck_dim": 4, "score_bottleneck_dim": 4},
        "data": {"cache_dir": "", "splits": ["train"],
                 "high_consensus_only": False, "magnitude_weighting": False,
                 "num_workers": 0},
        "optim": {"optimizer": "adamw", "lr": 1e-2, "weight_decay": 0.0,
                  "betas": (0.9, 0.999), "grad_clip": 1.0,
                  "warmup_steps": 1, "scheduler": "constant"},
    })

    fake_pairs = [_pair(i, score_for_chosen=1.0, score_for_rejected=-1.0)
                  for i in range(20)]

    with patch("natscore.train.online_trainer.WhisperFeatureExtractor", _FakeExtractor), \
         patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(fake_pairs)):
        # Import after patching so the class lookup hits the fake.
        from natscore.train.online_trainer import OnlineTrainer
        trainer = OnlineTrainer(cfg, device="cpu", token="dummy",
                                steps_per_epoch_hint=10, amp=False)
        assert trainer.model.trainable_param_count() > 0
        history = trainer.fit()

    assert len(history) == 5
    # Loss should be lower at the end than at the start when the model can
    # trivially separate (chosen > rejected by construction).
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["accuracy"] == 1.0


def test_online_trainer_checkpoint_roundtrip(tmp_path: Path):
    cfg = TrainConfig.from_dict({
        "run_name": "online-ckpt", "seed": 0, "batch_size": 2,
        "epochs": 1, "max_steps": 3, "log_every_steps": 1,
        "checkpoint_every_steps": 0,
        "output_dir": str(tmp_path), "wandb_enabled": False,
        "model": {"n_hidden_states": 3, "hidden_dim": 8,
                  "pooler_bottleneck_dim": 4, "score_bottleneck_dim": 4},
        "data": {"cache_dir": "", "splits": ["train"],
                 "high_consensus_only": False, "magnitude_weighting": False,
                 "num_workers": 0},
        "optim": {"optimizer": "adamw", "lr": 1e-2, "weight_decay": 0.0,
                  "betas": (0.9, 0.999), "grad_clip": 1.0,
                  "warmup_steps": 1, "scheduler": "constant"},
    })
    fake_pairs = [_pair(i, 1.0, -1.0) for i in range(10)]

    with patch("natscore.train.online_trainer.WhisperFeatureExtractor", _FakeExtractor), \
         patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter(fake_pairs)):
        from natscore.train.online_trainer import OnlineTrainer
        trainer = OnlineTrainer(cfg, device="cpu", token="dummy",
                                steps_per_epoch_hint=5, amp=False)
        trainer.fit()
        ckpt_path = trainer.save_checkpoint("rt")
        assert ckpt_path.exists()

        # New trainer, same config; load and confirm step counter survives.
        trainer2 = OnlineTrainer(cfg, device="cpu", token="dummy",
                                 steps_per_epoch_hint=5, amp=False)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.global_step == trainer.global_step
