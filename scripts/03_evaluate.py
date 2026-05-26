"""Milestone 4 entry: evaluate a NatScoreHead checkpoint on a cached eval set.

Usage:
    python scripts/03_evaluate.py \\
        --checkpoint outputs/natscore-small-v0/final.pt \\
        --cache cache/whisper_small_dev \\
        --label "natscore-small-v0 on dev[:100]"

Writes:
  - JSON results next to the checkpoint
  - A markdown row appended to docs/BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from natscore.eval.calibration import expected_calibration_error  # noqa: E402
from natscore.eval.speechjudge_eval import evaluate  # noqa: E402
from natscore.model import NatScoreHead, NatScoreHeadConfig  # noqa: E402
from natscore.train.config import TrainConfig  # noqa: E402

BENCHMARK_MD = REPO_ROOT / "docs" / "BENCHMARK.md"


def _load_model(checkpoint: Path, device: torch.device) -> NatScoreHead:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    if cfg_dict:
        train_cfg = TrainConfig.from_dict(cfg_dict)
        model_cfg = NatScoreHeadConfig(**train_cfg.to_dict()["model"])
    else:
        model_cfg = NatScoreHeadConfig()
    model = NatScoreHead(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _rebuild_margins_correct(model, cache_dir: Path, batch_size: int, device: torch.device):
    """Second pass for ECE: we want raw margins, not just per-pair correctness."""
    from torch.utils.data import DataLoader
    from natscore.data.pair_dataset import PairDataset, collate_pairs
    from natscore.eval.speechjudge_eval import _load_pair_meta

    ds = PairDataset(cache_dir=cache_dir, pair_meta=_load_pair_meta(cache_dir),
                     drop_pairs_missing_side=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_pairs)
    margins, correct = [], []
    with torch.inference_mode():
        for batch in loader:
            s_c = model(batch["feat_chosen"].to(device))
            s_r = model(batch["feat_rejected"].to(device))
            delta = (s_c - s_r).cpu().numpy()
            margins.extend(delta.tolist())
            correct.extend((delta > 0).astype(np.int64).tolist())
    return np.asarray(margins), np.asarray(correct)


def _append_benchmark_row(label: str, result_dict: dict, ece: float, ckpt_name: str) -> None:
    BENCHMARK_MD.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# NatScore — Benchmark\n\n"
        "> Headline metric: pairwise accuracy on SpeechJudge-Eval. Target "
        ">70% (beats half the field), stretch >73% (beats SpeechJudge-BTRM, "
        "the closest comparable). Aspirational >77% (matches SpeechJudge-GRM "
        "with ~1/14000th of its trainable parameters).\n\n"
        "| Run | Checkpoint | n_pairs | Pairwise acc | 95% CI | Mean margin | ECE |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    row = (
        f"| {label} | `{ckpt_name}` | {result_dict['n_pairs']} | "
        f"{100*result_dict['pairwise_accuracy']:.2f}% | "
        f"[{100*result_dict['ci_low']:.2f}, {100*result_dict['ci_high']:.2f}] | "
        f"{result_dict['mean_margin']:+.3f} | {100*ece:.2f}% |\n"
    )
    if not BENCHMARK_MD.exists() or "Pairwise acc" not in BENCHMARK_MD.read_text():
        BENCHMARK_MD.write_text(header + row)
    else:
        with BENCHMARK_MD.open("a") as fh:
            fh.write(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path,
                        help="Feature cache dir with manifest.jsonl and pair_meta.json.")
    parser.add_argument("--label", default=None,
                        help="Row label for docs/BENCHMARK.md. Default: checkpoint stem + cache stem.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--high-consensus-only", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = _load_model(args.checkpoint, device)
    print(f"  trainable params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"Evaluating on cache: {args.cache}")
    result = evaluate(
        model, args.cache,
        batch_size=args.batch_size, device=device,
        high_consensus_only=args.high_consensus_only,
    )
    margins, correct = _rebuild_margins_correct(model, args.cache, args.batch_size, device)
    cal = expected_calibration_error(margins, correct, n_bins=10)

    print(f"\n=== Results ===")
    print(f"  n_pairs:           {result.n_pairs}")
    print(f"  pairwise accuracy: {100*result.pairwise_accuracy:.2f}%  "
          f"(95% CI {100*result.ci_low:.2f}-{100*result.ci_high:.2f})")
    print(f"  mean margin:       {result.mean_margin:+.3f}")
    print(f"  ECE (10 bins):     {100*cal.ece:.2f}%")
    print(f"\nPer-subset accuracy:")
    for k, v in result.per_subset.items():
        print(f"  {k or '<empty>':>12s}  n={v['n_pairs']:>4d}  acc={100*v['accuracy']:.2f}%")
    print(f"\nPer-language accuracy:")
    for k, v in result.per_language.items():
        print(f"  {k or '<empty>':>12s}  n={v['n_pairs']:>4d}  acc={100*v['accuracy']:.2f}%")

    label = args.label or f"{args.checkpoint.stem} on {args.cache.name}"
    out = {
        "label": label,
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "result": result.as_dict(),
        "calibration": asdict(cal),
    }
    json_path = args.json_out or (args.checkpoint.parent / "eval_results.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote JSON results -> {json_path}")

    _append_benchmark_row(label, result.as_dict(), cal.ece, args.checkpoint.name)
    print(f"Appended row to {BENCHMARK_MD}")


if __name__ == "__main__":
    main()
