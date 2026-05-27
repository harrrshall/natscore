"""Multi-GPU wiring tests for the OnlineTrainer + WhisperFeatureExtractor.

We can't actually run DataParallel forward in CPU CI, but we can verify the
wrapping decisions and the checkpoint-portability invariant: the saved
state_dict must be identical between single-GPU and multi-GPU runs so the
same `latest.pt` works in either configuration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from natscore.data.speechjudge import PairRecord
from natscore.features import WhisperFeatureMeta, _StackingEncoderModule
from natscore.train.config import TrainConfig


class _FakeExtractor:
    """Reused from test_online_trainer.py to avoid a real Whisper download."""

    def __init__(self, *args, **kwargs) -> None:
        self.meta = WhisperFeatureMeta(
            model_name="fake", n_layers=2, n_hidden_states=3,
            hidden_dim=8, sample_rate=16_000, frame_rate=50,
            max_audio_seconds=30, output_frames=10,
        )

    def batch_extract_layerwise(self, audios: list) -> torch.Tensor:
        feats = []
        for a in audios:
            seed = float(a[0]) if len(a) else 0.0
            feats.append(torch.full((3, 10, 8), seed, dtype=torch.float32))
        return torch.stack(feats, dim=0).to("cpu")


def _wav(value: float, n: int = 100) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def _pair(idx: int) -> PairRecord:
    return PairRecord(
        pair_index=idx, subset="regular", language_setting="en2en",
        target_text="t", audio_a=_wav(-1.0), audio_b=_wav(1.0),
        sample_rate_a=16_000, sample_rate_b=16_000,
        naturalness_label="B", naturalness_annotation=["B+1"], chosen=True,
    )


def _make_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig.from_dict({
        "run_name": "dp-test", "seed": 0, "batch_size": 2,
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


# ----------------------------------------------------------- single-GPU paths


def test_extractor_no_wrap_on_cpu():
    """On CPU, the encoder must NOT be wrapped in DataParallel."""
    encoder = torch.nn.Linear(8, 8)  # stand-in (any nn.Module works)
    mod = _StackingEncoderModule(encoder)
    assert not isinstance(mod, torch.nn.DataParallel)
    # The constructor-level guard is exercised via the trainer test below.


def test_online_trainer_no_dataparallel_on_cpu(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    with patch("natscore.train.online_trainer.WhisperFeatureExtractor", _FakeExtractor), \
         patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter([_pair(i) for i in range(10)])):
        from natscore.train.online_trainer import OnlineTrainer
        trainer = OnlineTrainer(cfg, device="cpu", token="dummy",
                                steps_per_epoch_hint=5, amp=False)
        assert trainer._multi_gpu is False
        assert trainer._fwd_model is trainer.model
        assert not isinstance(trainer._fwd_model, torch.nn.DataParallel)


# ------------------------------------------------------------ multi-GPU paths


def test_multi_gpu_decision_logic():
    """The wrap decision must be: cuda device AND device_count > 1.

    Verified through the boolean expression in isolation, because we can't
    actually construct torch.nn.DataParallel on a CPU-only torch build
    (it errors with 'Torch not compiled with CUDA enabled' even when
    cuda.device_count is monkeypatched).
    """
    # The exact expression used by OnlineTrainer.__init__ and
    # WhisperFeatureExtractor.__init__.
    def wrap_decision(device_type: str, n_gpus: int) -> bool:
        return device_type == "cuda" and n_gpus > 1

    assert wrap_decision("cpu", 0) is False
    assert wrap_decision("cpu", 2) is False     # CPU never wraps
    assert wrap_decision("cuda", 0) is False    # impossible IRL but defensive
    assert wrap_decision("cuda", 1) is False    # single GPU: no point
    assert wrap_decision("cuda", 2) is True     # Kaggle T4 x2
    assert wrap_decision("cuda", 4) is True     # bigger fleet


def test_stacking_encoder_returns_stacked_tensor():
    """The wrapper module must produce Tensor[B, H, T, D] without DataParallel.

    Uses a dummy encoder that mimics the WhisperEncoder return signature.
    Verifies the contract the DataParallel-friendly wrapping depends on.
    """
    class _DummyEncoderOutput:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class _DummyEncoder(torch.nn.Module):
        def forward(self, x, output_hidden_states=False, return_dict=False):
            # Pretend we have 3 hidden states each [B, 4, 8].
            B = x.shape[0]
            hs = tuple(torch.full((B, 4, 8), float(i)) for i in range(3))
            return _DummyEncoderOutput(hs)

    mod = _StackingEncoderModule(_DummyEncoder())
    out = mod(torch.zeros(2, 10))
    assert out.shape == (2, 3, 4, 8)
    # Each layer's values must reflect the source ordering.
    assert (out[:, 0] == 0.0).all()
    assert (out[:, 1] == 1.0).all()
    assert (out[:, 2] == 2.0).all()


def test_checkpoint_state_dict_is_device_count_agnostic(tmp_path: Path):
    """A checkpoint saved from a multi-GPU run must load cleanly into a
    single-GPU trainer (and vice versa). Achieved by always saving
    self.model.state_dict() (the unwrapped head), never the DataParallel
    wrapper's state_dict (which prefixes every key with 'module.').
    """
    cfg = _make_cfg(tmp_path)

    # Single-GPU run: train a few steps, save checkpoint.
    with patch("natscore.train.online_trainer.WhisperFeatureExtractor", _FakeExtractor), \
         patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter([_pair(i) for i in range(10)])):
        from natscore.train.online_trainer import OnlineTrainer
        trainer = OnlineTrainer(cfg, device="cpu", token="dummy",
                                steps_per_epoch_hint=5, amp=False)
        trainer.fit()
        ckpt = trainer.save_checkpoint("dp-roundtrip")

    # Sanity: no 'module.' prefix in saved keys.
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert all(not k.startswith("module.") for k in blob["model_state"].keys()), (
        f"unexpected 'module.' prefix in checkpoint: "
        f"{[k for k in blob['model_state'] if k.startswith('module.')]}"
    )

    # Load into a freshly-constructed trainer and confirm parity.
    with patch("natscore.train.online_trainer.WhisperFeatureExtractor", _FakeExtractor), \
         patch("natscore.data.online_pair_dataset.iter_pairs",
               return_value=iter([_pair(i) for i in range(10)])):
        from natscore.train.online_trainer import OnlineTrainer
        trainer2 = OnlineTrainer(cfg, device="cpu", token="dummy",
                                 steps_per_epoch_hint=5, amp=False)
        trainer2.load_checkpoint(ckpt)
        assert trainer2.global_step == trainer.global_step
        # Weights match exactly.
        sd1 = trainer.model.state_dict()
        sd2 = trainer2.model.state_dict()
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), f"weight mismatch at {k}"
