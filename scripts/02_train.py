"""Milestone 3 entry point: train the NatScore head on cached Whisper features.

Usage:
    python scripts/02_train.py --config configs/train.small.yaml
    python scripts/02_train.py --config configs/train.sanity.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from natscore.train.config import TrainConfig  # noqa: E402
from natscore.train.trainer import Trainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="YAML config (see configs/train.small.yaml).")
    parser.add_argument("--device", default=None,
                        help="Override device, e.g. cpu / cuda / cuda:0.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging for this run.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override config.max_steps (handy for smoke runs).")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint .pt to resume from.")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.no_wandb:
        cfg.wandb_enabled = False
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps

    trainer = Trainer(cfg, device=args.device)
    print(f"Trainable params: {trainer.model.trainable_param_count():,}")
    print(f"Dataset size: {len(trainer.dataset)} pairs; "
          f"batches/epoch: {len(trainer.loader)}; "
          f"total steps: {trainer.total_steps}")
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.global_step}")
    trainer.fit()


if __name__ == "__main__":
    main()
