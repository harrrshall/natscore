"""Hyperparameter dataclasses for NatScore training.

YAML-roundtrippable. One config object end-to-end so training is
exactly reproducible from the saved config alongside each checkpoint
(per PROJECT_PLAN.md s6.2 reproducibility requirements).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    n_hidden_states: int = 13
    hidden_dim: int = 768
    pooler_bottleneck_dim: int = 256
    score_bottleneck_dim: int = 256
    dropout: float = 0.0
    init_layer_weights: str = "uniform"


@dataclass
class DataConfig:
    cache_dir: str = "cache/whisper_small"
    splits: list[str] = field(default_factory=lambda: ["train"])
    high_consensus_only: bool = False        # filter pairs to chosen==True
    magnitude_weighting: bool = False        # use parse_magnitude_weight
    drop_pairs_missing_side: bool = True
    num_workers: int = 0                     # 0 means main process (safer with memmap)


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    scheduler: str = "cosine"                # "constant" | "cosine" | "linear"


@dataclass
class TrainConfig:
    run_name: str = "natscore-dev"
    seed: int = 42
    batch_size: int = 8
    epochs: int = 5
    max_steps: int | None = None             # overrides epochs when set
    eval_every_steps: int = 0                # 0 = no in-run eval
    checkpoint_every_steps: int = 500
    log_every_steps: int = 10
    output_dir: str = "outputs"
    wandb_project: str | None = "natscore"
    wandb_entity: str | None = None
    wandb_enabled: bool = True

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # ----------------------------------------------------------------- I/O

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        # Pull nested sub-dataclasses out so we can pass them via field defaults.
        kwargs: dict[str, Any] = {}
        nested = {"model": ModelConfig, "data": DataConfig, "optim": OptimConfig}
        for key, sub_cls in nested.items():
            sub_raw = raw.pop(key, None) or {}
            # YAML loads tuples as lists; coerce back so equality / optimizer
            # APIs that prefer tuples (e.g. AdamW.betas) behave consistently.
            if sub_cls is OptimConfig and "betas" in sub_raw:
                sub_raw["betas"] = tuple(sub_raw["betas"])
            kwargs[key] = sub_cls(**sub_raw)
        for f in fields(cls):
            if f.name in nested:
                continue
            if f.name in raw:
                kwargs[f.name] = raw[f.name]
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
