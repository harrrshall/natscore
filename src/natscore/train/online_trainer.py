"""Online-encoder Bradley-Terry training loop for NatScoreHead.

PROJECT_PLAN.md s6.1.1 path B: the full-cache pre-extraction is infeasible
(5.34 TB). Instead, run the frozen Whisper encoder forward at training
time on GPU and backprop only through the head. This trades
~recomputation-per-epoch for the ability to train on the full ~42K split.

Resumability:

- The trainer state is fully captured by (model_state, optimizer_state,
  scheduler_state, global_step, samples_seen, config). Persisted to
  outputs/<run_name>/step_NNNNNN.pt and outputs/<run_name>/latest.pt.
- The streaming dataset is shard-deterministic given (shard_seed, epoch).
  After a crash, resuming from latest.pt + starting from epoch =
  ckpt['epoch'] re-walks the shard order from the right place. We do
  NOT try to pick up mid-shard -- shard granularity is good enough.
- Pre-emptible Kaggle sessions can save every N steps; a fresh kernel
  loads the latest checkpoint and continues.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from natscore.data.online_pair_dataset import StreamingPairDataset, collate_streaming_pairs
from natscore.features import WhisperFeatureExtractor
from natscore.model import NatScoreHead, NatScoreHeadConfig
from natscore.train.config import TrainConfig
from natscore.train.losses import bradley_terry_loss
from natscore.train.trainer import _maybe_init_wandb, _make_scheduler, set_seed


class OnlineTrainer:
    """BT training with on-the-fly Whisper encoder forward.

    Differences vs `Trainer` (cache-based):
      - Builds a WhisperFeatureExtractor on the same device.
      - Uses StreamingPairDataset (IterableDataset) instead of PairDataset.
      - Each train step: extractor forward (no grad) -> head forward (grad).
      - Mixed-precision optional via torch.autocast.
      - Resume protocol writes outputs/<run>/latest.pt every checkpoint.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        device: str | torch.device | None = None,
        token: str | None = None,
        split: str = "train",
        steps_per_epoch_hint: int | None = None,
        amp: bool = True,
        encoder_model: str = "openai/whisper-small",
    ) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.amp = amp and self.device.type == "cuda"
        self.split = split
        self.token = token

        # ----- encoder (frozen)
        # Encoder is in fp32 even with AMP -- autocast picks fp16 per-op
        # during the forward; storing weights in fp16 is a different
        # optimization that we skip for safety.
        self.extractor = WhisperFeatureExtractor(
            model_name=encoder_model, device=self.device, dtype=torch.float32,
        )

        # ----- head (trainable)
        model_cfg = NatScoreHeadConfig(**asdict(cfg.model))
        self.model = NatScoreHead(model_cfg).to(self.device)

        # ----- dataset / loader
        self.dataset = StreamingPairDataset(
            split=split,
            limit=None,                         # full split unless caller overrides
            token=token,
            high_consensus_only=cfg.data.high_consensus_only,
            magnitude_weighting=cfg.data.magnitude_weighting,
            subset_filter=cfg.data.subset_filter,
            shuffle_shards=True,
            shard_seed=cfg.seed,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,                      # streaming dataset; no torch shuffle
            num_workers=cfg.data.num_workers,
            collate_fn=collate_streaming_pairs,
        )

        # ----- optimizer / scheduler / amp
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
            betas=cfg.optim.betas,
        )
        # Without knowing the dataset length up front (it's streaming),
        # the user can pass an estimate; else fall back to a long horizon.
        self.steps_per_epoch_hint = steps_per_epoch_hint or 5_000
        self.total_steps = cfg.max_steps or (cfg.epochs * self.steps_per_epoch_hint)
        self.scheduler = _make_scheduler(self.optimizer, cfg, self.total_steps)
        self.scaler = torch.amp.GradScaler("cuda") if self.amp else None

        # ----- output / wandb
        self.global_step = 0
        self.samples_seen = 0
        self.output_dir = Path(cfg.output_dir) / cfg.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(self.output_dir / "config.yaml")
        self._wandb = _maybe_init_wandb(cfg)
        self._loss_history: list[dict] = []

    # ----------------------------------------------------------------- step

    def _encode_batch(self, wavs: list[np.ndarray]) -> torch.Tensor:
        """Run the frozen encoder on a list of waveforms; returns [B, H, T, D]."""
        # No autocast around the encoder call: mel preprocessing inside
        # WhisperFeatureExtractor allocates tensors itself; mixing AMP
        # contexts is fragile. We let the encoder run in its native dtype
        # and rely on torch's free fp16 matmul on supported GPUs.
        return self.extractor.batch_extract_layerwise(wavs)

    def _step(self, batch: dict) -> dict[str, float]:
        wavs_c: list[np.ndarray] = batch["wav_chosen"]
        wavs_r: list[np.ndarray] = batch["wav_rejected"]
        weight = batch["weight"].to(self.device, non_blocking=True)

        # Encoder forward (no grad, no autocast on the extractor itself).
        feat_c = self._encode_batch(wavs_c)
        feat_r = self._encode_batch(wavs_r)

        if self.amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                s_c = self.model(feat_c)
                s_r = self.model(feat_r)
                out = bradley_terry_loss(s_c, s_r, weight=weight)
        else:
            s_c = self.model(feat_c)
            s_r = self.model(feat_r)
            out = bradley_terry_loss(s_c, s_r, weight=weight)

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(out.loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out.loss.backward()
            if self.cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.optimizer.step()
        self.scheduler.step()

        self.samples_seen += len(wavs_c)
        return {
            "loss": float(out.loss.detach().cpu()),
            "accuracy": float(out.accuracy.detach().cpu()),
            "mean_margin": float(out.mean_margin.detach().cpu()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "samples_seen": self.samples_seen,
        }

    # ----------------------------------------------------------------- fit

    def fit(self, resume_from: str | Path | None = None) -> list[dict]:
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            print(f"Resumed from {resume_from} at step {self.global_step}")

        cfg = self.cfg
        t0 = time.time()
        start_epoch = self.global_step // max(1, self.steps_per_epoch_hint)
        for epoch in range(start_epoch, cfg.epochs):
            self.dataset.set_epoch(epoch)
            for batch in self.loader:
                metrics = self._step(batch)
                self.global_step += 1
                metrics["step"] = self.global_step
                metrics["epoch"] = epoch
                self._loss_history.append(metrics)

                if self.global_step % cfg.log_every_steps == 0:
                    print(
                        f"step={self.global_step:6d} epoch={epoch:2d} "
                        f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
                        f"margin={metrics['mean_margin']:+.3f} lr={metrics['lr']:.2e} "
                        f"seen={metrics['samples_seen']:>6d}"
                    )
                    if self._wandb is not None:
                        self._wandb.log(metrics, step=self.global_step)

                if (cfg.checkpoint_every_steps > 0
                        and self.global_step % cfg.checkpoint_every_steps == 0):
                    self.save_checkpoint(f"step_{self.global_step:06d}")
                    self.save_checkpoint("latest")  # for resume convenience

                if cfg.max_steps and self.global_step >= cfg.max_steps:
                    break
            if cfg.max_steps and self.global_step >= cfg.max_steps:
                break

        self.save_checkpoint("final")
        self.save_checkpoint("latest")
        elapsed = time.time() - t0
        print(f"Training done. {self.global_step} steps in {elapsed:.1f}s "
              f"({elapsed / max(1, self.global_step):.2f} s/step). "
              f"Samples seen: {self.samples_seen}.")
        if self._wandb is not None:
            self._wandb.finish()
        return self._loss_history

    # --------------------------------------------------------------- io

    def save_checkpoint(self, tag: str) -> Path:
        path = self.output_dir / f"{tag}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict() if self.scaler else None,
                "global_step": self.global_step,
                "samples_seen": self.samples_seen,
                "config": self.cfg.to_dict(),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler is not None and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = ckpt.get("global_step", 0)
        self.samples_seen = ckpt.get("samples_seen", 0)
