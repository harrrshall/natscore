"""Unit tests for the TrainConfig dataclass + YAML round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from natscore.train.config import DataConfig, ModelConfig, OptimConfig, TrainConfig


def test_defaults_construct_cleanly():
    cfg = TrainConfig()
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.optim, OptimConfig)
    assert cfg.seed == 42


def test_from_dict_overrides_defaults():
    cfg = TrainConfig.from_dict({
        "seed": 7, "batch_size": 16,
        "model": {"dropout": 0.3, "init_layer_weights": "balanced"},
        "data":  {"cache_dir": "cache/somewhere"},
        "optim": {"lr": 5e-4, "scheduler": "linear"},
    })
    assert cfg.seed == 7
    assert cfg.batch_size == 16
    assert cfg.model.dropout == 0.3
    assert cfg.model.init_layer_weights == "balanced"
    assert cfg.data.cache_dir == "cache/somewhere"
    assert cfg.optim.lr == pytest.approx(5e-4)
    assert cfg.optim.scheduler == "linear"


def test_yaml_roundtrip(tmp_path: Path):
    cfg = TrainConfig.from_dict({"run_name": "x", "batch_size": 11})
    p = tmp_path / "c.yaml"
    cfg.save(p)
    raw = yaml.safe_load(p.read_text())
    assert raw["run_name"] == "x"
    assert raw["batch_size"] == 11
    cfg2 = TrainConfig.from_yaml(p)
    assert cfg2.to_dict() == cfg.to_dict()


def test_to_dict_is_full_tree():
    cfg = TrainConfig()
    d = cfg.to_dict()
    for key in ("run_name", "seed", "batch_size", "model", "data", "optim"):
        assert key in d
    assert "lr" in d["optim"]
    assert "cache_dir" in d["data"]
