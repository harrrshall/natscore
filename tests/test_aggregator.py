"""Unit tests for the M5b benchmark aggregator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
# import-by-filename via importlib so the leading "05_" digit doesn't bite us.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "aggregator", SCRIPTS_DIR / "05_aggregate_benchmark.py"
)
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def _make_eval_json(path: Path, run_name: str, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ablation_config": "train.kaggle",
        "run_name": run_name,
        "n_pairs": 6000,
        "pairwise_accuracy": accuracy,
        "mean_margin": 1.5,
        "ci_low": accuracy - 0.05,
        "ci_high": accuracy + 0.05,
        "ece": 0.10,
        "per_subset": {},
        "per_language": {},
    }))


def test_gather_picks_up_eval_dev_json(tmp_path: Path):
    _make_eval_json(tmp_path / "run_a" / "eval_dev.json", "run_a", 0.72)
    _make_eval_json(tmp_path / "run_b" / "eval_dev.json", "run_b", 0.65)
    rows = agg._gather(tmp_path)
    assert len(rows) == 2
    names = {r["run_name"] for r in rows}
    assert names == {"run_a", "run_b"}


def test_gather_handles_legacy_eval_results_json(tmp_path: Path):
    # Old schema (from scripts/03_evaluate.py)
    legacy = tmp_path / "old_run" / "eval_results.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({
        "label": "legacy / dev",
        "checkpoint": "x.pt",
        "cache": "x",
        "result": {
            "n_pairs": 100, "pairwise_accuracy": 0.52,
            "mean_margin": 0.13, "ci_low": 0.43, "ci_high": 0.62,
            "per_subset": {}, "per_language": {},
        },
        "calibration": {"ece": 0.31},
    }))
    rows = agg._gather(tmp_path)
    assert len(rows) == 1
    assert rows[0]["pairwise_accuracy"] == 0.52
    assert rows[0]["ece"] == 0.31


def test_build_markdown_sorts_by_accuracy_desc(tmp_path: Path):
    _make_eval_json(tmp_path / "low" / "eval_dev.json", "low_run", 0.55)
    _make_eval_json(tmp_path / "hi"  / "eval_dev.json", "hi_run",  0.75)
    rows = agg._gather(tmp_path)
    md = agg._build_markdown(rows)
    lines = md.splitlines()
    # The first row in the table body should be the highest-accuracy one.
    body = [ln for ln in lines if ln.startswith("|") and "Run" not in ln and "---" not in ln]
    assert "hi_run" in body[0]
    assert "low_run" in body[1]


def test_splice_section_replaces_existing(tmp_path: Path):
    existing = (
        "# NatScore — Benchmark\n\n"
        "Header text.\n\n"
        "## Ablations (M5b)\n\nOLD STUFF\n\n"
        "## After\n\ntrailing.\n"
    )
    new = "## Ablations (M5b)\n\nNEW STUFF\n"
    out = agg._splice_section(existing, new)
    assert "OLD STUFF" not in out
    assert "NEW STUFF" in out
    assert "## After" in out             # other sections preserved


def test_splice_section_appends_when_missing(tmp_path: Path):
    existing = "# NatScore — Benchmark\n\nHeader text.\n"
    new = "## Ablations (M5b)\n\nfresh\n"
    out = agg._splice_section(existing, new)
    assert out.endswith("fresh\n")
    assert "Header text." in out
