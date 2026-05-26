"""Pairwise-accuracy evaluation on a SpeechJudge feature cache.

PROJECT_PLAN.md s8 Milestone 4 headline metric: pairwise accuracy on
SpeechJudge-Eval. Target >70% (beats half the field), stretch >73%
(beats SpeechJudge-BTRM). Aspirational >77% (matches the 7B GRM).

Workflow:

    eval_cache_dir/                       # produced by 01_extract_features.py
        features/...
        manifest.jsonl
        pair_meta.json                    # produced by build_pair_meta.py

    >>> from natscore.eval.speechjudge_eval import evaluate
    >>> result = evaluate(model, eval_cache_dir)
    >>> result.pairwise_accuracy           # scalar
    >>> result.confidence_intervals        # 95% CI via bootstrap
    >>> result.per_subset                  # {subset -> accuracy}
    >>> result.per_language                # {language_setting -> accuracy}

Implementation notes:

- The model is run in inference_mode (no autograd) and forced to eval()
  so dropout / training-time noise doesn't leak in.
- Per-subset / per-language breakdowns are bookkeeping over the same
  forward passes (no extra compute) -- they fall straight out of the
  manifest metadata.
- Confidence intervals use 1K bootstrap replicates on the per-pair
  correctness vector. Fast and standard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from natscore.data.feature_cache import FeatureCache
from natscore.data.pair_dataset import PairDataset, PairMeta, collate_pairs


@dataclass
class EvalResult:
    n_pairs: int
    pairwise_accuracy: float
    mean_margin: float
    ci_low: float                          # 95% bootstrap lower bound
    ci_high: float                         # 95% bootstrap upper bound
    per_subset: dict[str, dict[str, float]] = field(default_factory=dict)
    per_language: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "pairwise_accuracy": self.pairwise_accuracy,
            "mean_margin": self.mean_margin,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "per_subset": self.per_subset,
            "per_language": self.per_language,
        }


def _bootstrap_ci(correct: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    samples = rng.choice(correct, size=(n_boot, n), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def _load_pair_meta(cache_dir: Path) -> dict[int, PairMeta]:
    """Read the pair_meta.json sidecar."""
    import json

    p = cache_dir / "pair_meta.json"
    if not p.exists():
        raise FileNotFoundError(
            f"pair_meta.json missing at {p}. "
            "Run scripts/build_pair_meta.py with matching --split/--limit."
        )
    with p.open() as fh:
        raw = json.load(fh)
    return {int(k): PairMeta(**v) for k, v in raw.items()}


def evaluate(
    model: torch.nn.Module,
    cache_dir: str | Path,
    *,
    batch_size: int = 8,
    device: str | torch.device | None = None,
    high_consensus_only: bool = False,
    n_bootstrap: int = 1000,
) -> EvalResult:
    """Pairwise-accuracy evaluation against a feature cache.

    Args:
        model: trained NatScoreHead (or anything with the same forward signature).
        cache_dir: directory produced by 01_extract_features.py + build_pair_meta.py.
        batch_size: forward batch size.
        device: 'cpu' | 'cuda' | None (auto-detect).
        high_consensus_only: if True, restrict to pairs with `chosen == True`.
        n_bootstrap: number of bootstrap replicates for the 95% CI.
    """
    cache_dir = Path(cache_dir)
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pair_meta = _load_pair_meta(cache_dir)
    dataset = PairDataset(
        cache_dir=cache_dir,
        pair_meta=pair_meta,
        high_consensus_only=high_consensus_only,
        magnitude_weighting=False,
        drop_pairs_missing_side=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No pairs to evaluate. Cache at {cache_dir} produced 0 usable "
            "pair_meta join hits."
        )

    # Per-pair manifest entries needed for the subset / language breakdowns.
    cache = FeatureCache(cache_dir)
    chosen_meta: dict[int, dict] = {}
    for entry in cache.iter_manifest():
        # entry.pair_index may appear twice (A and B); both share subset / language.
        chosen_meta[entry.pair_index] = {
            "subset": entry.subset,
            "language_setting": entry.language_setting,
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pairs,
    )

    model.eval().to(device)

    correct: list[int] = []
    margins: list[float] = []
    pair_indices: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            feat_c = batch["feat_chosen"].to(device, non_blocking=True)
            feat_r = batch["feat_rejected"].to(device, non_blocking=True)
            s_c = model(feat_c)
            s_r = model(feat_r)
            delta = (s_c - s_r).detach().cpu().numpy()
            margins.extend(delta.tolist())
            correct.extend((delta > 0).astype(np.int64).tolist())
            pair_indices.extend(batch["pair_index"].tolist())

    correct_arr = np.asarray(correct, dtype=np.float64)
    accuracy = float(correct_arr.mean()) if len(correct_arr) else float("nan")
    mean_margin = float(np.mean(margins)) if margins else float("nan")
    ci_low, ci_high = _bootstrap_ci(correct_arr, n_boot=n_bootstrap)

    # Breakdowns
    per_subset_counts: dict[str, list[int]] = {}
    per_language_counts: dict[str, list[int]] = {}
    for pi, c in zip(pair_indices, correct):
        meta = chosen_meta.get(pi, {})
        per_subset_counts.setdefault(meta.get("subset", ""), []).append(c)
        per_language_counts.setdefault(meta.get("language_setting", ""), []).append(c)

    def _summarize(d: dict[str, list[int]]) -> dict[str, dict[str, float]]:
        out = {}
        for k, vals in sorted(d.items()):
            arr = np.asarray(vals)
            out[k] = {
                "n_pairs": int(arr.size),
                "accuracy": float(arr.mean()) if arr.size else float("nan"),
            }
        return out

    return EvalResult(
        n_pairs=len(correct),
        pairwise_accuracy=accuracy,
        mean_margin=mean_margin,
        ci_low=ci_low,
        ci_high=ci_high,
        per_subset=_summarize(per_subset_counts),
        per_language=_summarize(per_language_counts),
    )
