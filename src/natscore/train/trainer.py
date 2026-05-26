"""Bradley-Terry training loop for NatScoreHead.

Loads pre-extracted Whisper features from a FeatureCache, joins them
into pairs, runs the head, applies the BT loss, optimises. Built around
the small training-from-cache iteration path (PROJECT_PLAN.md s6.1.1);
online encoder extraction is a future addition for full Kaggle runs.

Reproducibility (s6.2): all seeds set; full TrainConfig saved next to
every checkpoint; deterministic torch ops where free.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from natscore.data.pair_dataset import PairDataset, collate_pairs
from natscore.model import NatScoreHead, NatScoreHeadConfig
from natscore.train.config import TrainConfig
from natscore.train.losses import bradley_terry_loss


# --------------------------------------------------------------- determinism


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN convolutions: deterministic is slower but reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------- W&B


def _maybe_init_wandb(cfg: TrainConfig):
    if not cfg.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; logging disabled. `pip install wandb` to enable.")
        return None
    if os.environ.get("WANDB_DISABLED") == "true":
        return None
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config=cfg.to_dict(),
        reinit=True,
    )
    return run


# ----------------------------------------------------------------- scheduler


def _make_scheduler(optimizer, cfg: TrainConfig, total_steps: int):
    warmup = max(1, cfg.optim.warmup_steps)
    scheduler_name = cfg.optim.scheduler

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        if scheduler_name == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_name == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_name == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown scheduler: {scheduler_name!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_at)


# ------------------------------------------------------------------ trainer


class Trainer:
    def __init__(self, cfg: TrainConfig, *, device: str | torch.device | None = None) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model_cfg = NatScoreHeadConfig(**asdict(cfg.model))
        self.model = NatScoreHead(model_cfg).to(self.device)

        self.dataset = self._build_dataset()
        self.loader = self._build_loader()

        self.optimizer = self._build_optimizer()
        steps_per_epoch = max(1, len(self.loader))
        total_steps = cfg.max_steps or (cfg.epochs * steps_per_epoch)
        self.total_steps = total_steps
        self.scheduler = _make_scheduler(self.optimizer, cfg, total_steps)

        self.global_step = 0
        self.start_epoch = 0
        self.output_dir = Path(cfg.output_dir) / cfg.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(self.output_dir / "config.yaml")

        self._wandb = _maybe_init_wandb(cfg)
        self._loss_history: list[dict] = []

    # ------------------------------------------------------------ construction

    def _build_dataset(self) -> PairDataset:
        # Load the side-car pair_meta map if present; without it, the
        # PairDataset will still load but every pair gets skipped because
        # naturalness_label is unknown.
        cache_dir = Path(self.cfg.data.cache_dir)
        meta_path = cache_dir / "pair_meta.json"
        pair_meta: dict = {}
        if meta_path.exists():
            with meta_path.open() as fh:
                raw = json.load(fh)
            from natscore.data.pair_dataset import PairMeta
            pair_meta = {int(k): PairMeta(**v) for k, v in raw.items()}
        return PairDataset(
            cache_dir=cache_dir,
            pair_meta=pair_meta,
            high_consensus_only=self.cfg.data.high_consensus_only,
            magnitude_weighting=self.cfg.data.magnitude_weighting,
            drop_pairs_missing_side=self.cfg.data.drop_pairs_missing_side,
        )

    def _build_loader(self) -> DataLoader:
        if len(self.dataset) == 0:
            raise RuntimeError(
                f"PairDataset is empty. Did you cache features and write "
                f"{Path(self.cfg.data.cache_dir) / 'pair_meta.json'}? "
                "Run scripts/01_extract_features.py and the meta builder."
            )
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            collate_fn=collate_pairs,
            drop_last=False,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optim.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.cfg.optim.lr,
                weight_decay=self.cfg.optim.weight_decay,
                betas=self.cfg.optim.betas,
            )
        raise ValueError(f"Unknown optimizer: {self.cfg.optim.optimizer!r}")

    # ------------------------------------------------------------------ step

    def _step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        feat_c = batch["feat_chosen"].to(self.device, non_blocking=True)
        feat_r = batch["feat_rejected"].to(self.device, non_blocking=True)
        weight = batch["weight"].to(self.device, non_blocking=True)

        s_c = self.model(feat_c)
        s_r = self.model(feat_r)
        out = bradley_terry_loss(s_c, s_r, weight=weight)

        self.optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        if self.cfg.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": float(out.loss.detach().cpu()),
            "accuracy": float(out.accuracy.detach().cpu()),
            "mean_margin": float(out.mean_margin.detach().cpu()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ------------------------------------------------------------------ fit

    def fit(self) -> list[dict]:
        cfg = self.cfg
        t0 = time.time()
        for epoch in range(self.start_epoch, cfg.epochs):
            for batch in self.loader:
                metrics = self._step(batch)
                self.global_step += 1
                metrics["step"] = self.global_step
                metrics["epoch"] = epoch
                self._loss_history.append(metrics)

                if self.global_step % cfg.log_every_steps == 0:
                    print(
                        f"step={self.global_step:5d} epoch={epoch:2d} "
                        f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
                        f"margin={metrics['mean_margin']:+.3f} lr={metrics['lr']:.2e}"
                    )
                    if self._wandb is not None:
                        self._wandb.log(metrics, step=self.global_step)

                if (cfg.checkpoint_every_steps > 0
                        and self.global_step % cfg.checkpoint_every_steps == 0):
                    self.save_checkpoint(f"step_{self.global_step:06d}")

                if cfg.max_steps and self.global_step >= cfg.max_steps:
                    break
            if cfg.max_steps and self.global_step >= cfg.max_steps:
                break

        self.save_checkpoint("final")
        elapsed = time.time() - t0
        print(f"Training done. {self.global_step} steps in {elapsed:.1f}s "
              f"({elapsed / max(1, self.global_step):.2f} s/step).")
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
                "global_step": self.global_step,
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
        self.global_step = ckpt.get("global_step", 0)
