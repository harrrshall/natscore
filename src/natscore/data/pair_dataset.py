"""PyTorch Dataset over the FeatureCache that emits preference pairs.

Each item is one (chosen, rejected) pair of feature tensors plus a per-pair
weight derived from `naturalness_annotation`. The cache stores one
manifest entry per clip; this dataset joins entries by `pair_index`,
picks chosen vs rejected via `naturalness_label`, then loads the two
safetensors files when __getitem__ is called.

The cache manifest does NOT store `naturalness_label` or
`naturalness_annotation` (the cache writer in scripts/01_extract_features.py
only persists what FeatureCache.write accepts). So this dataset takes a
secondary `pair_meta` mapping at construction time, keyed by pair_index.
A helper `pair_meta_from_dataset(...)` streams the train split once to
build that map without re-downloading audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from natscore.data.feature_cache import CacheEntry, FeatureCache
from natscore.train.losses import parse_magnitude_weight


@dataclass(frozen=True)
class PairMeta:
    pair_index: int
    naturalness_label: str            # "A" or "B"
    naturalness_annotation: list[str]  # e.g. ["B+2", "B+1"]
    chosen: bool                       # high-rater-agreement flag


@dataclass(frozen=True)
class PairItem:
    pair_index: int
    feat_chosen: torch.Tensor          # [H, T, D]
    feat_rejected: torch.Tensor        # [H, T, D]
    weight: float                      # >= 0
    duration_chosen: float
    duration_rejected: float


class PairDataset(Dataset[PairItem]):
    """Joins clips by pair_index, applies optional consensus / weighting filters."""

    def __init__(
        self,
        cache_dir: str | Path,
        pair_meta: dict[int, PairMeta] | None = None,
        *,
        high_consensus_only: bool = False,
        magnitude_weighting: bool = False,
        drop_pairs_missing_side: bool = True,
    ) -> None:
        self.cache = FeatureCache(cache_dir)
        self.pair_meta: dict[int, PairMeta] = pair_meta or {}
        self.high_consensus_only = high_consensus_only
        self.magnitude_weighting = magnitude_weighting

        # Build per-pair (entry_a, entry_b) index.
        by_pair: dict[int, dict[str, CacheEntry]] = {}
        for entry in self.cache.iter_manifest():
            by_pair.setdefault(entry.pair_index, {})[entry.side] = entry

        self._pairs: list[tuple[CacheEntry, CacheEntry, float]] = []
        for idx in sorted(by_pair):
            sides = by_pair[idx]
            if "A" not in sides or "B" not in sides:
                if not drop_pairs_missing_side:
                    raise ValueError(
                        f"pair_index={idx} missing side(s); have {sorted(sides)}"
                    )
                continue
            entry_a, entry_b = sides["A"], sides["B"]

            meta = self.pair_meta.get(idx)
            if meta is None:
                # Without meta we cannot pick chosen vs rejected -- skip.
                continue
            if high_consensus_only and not meta.chosen:
                continue

            if meta.naturalness_label == "A":
                chosen, rejected = entry_a, entry_b
            elif meta.naturalness_label == "B":
                chosen, rejected = entry_b, entry_a
            else:
                continue

            if magnitude_weighting:
                w = parse_magnitude_weight(
                    meta.naturalness_annotation, meta.naturalness_label
                )
            else:
                w = 1.0
            self._pairs.append((chosen, rejected, w))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> PairItem:
        chosen_entry, rejected_entry, weight = self._pairs[idx]
        feat_chosen = self.cache.load(chosen_entry.clip_id).to(torch.float32)
        feat_rejected = self.cache.load(rejected_entry.clip_id).to(torch.float32)
        return PairItem(
            pair_index=chosen_entry.pair_index,
            feat_chosen=feat_chosen,
            feat_rejected=feat_rejected,
            weight=weight,
            duration_chosen=chosen_entry.duration_seconds,
            duration_rejected=rejected_entry.duration_seconds,
        )


def collate_pairs(batch: list[PairItem]) -> dict[str, torch.Tensor]:
    """torch DataLoader collate_fn. Returns a dict of stacked tensors."""
    return {
        "feat_chosen": torch.stack([b.feat_chosen for b in batch], dim=0),
        "feat_rejected": torch.stack([b.feat_rejected for b in batch], dim=0),
        "weight": torch.tensor([b.weight for b in batch], dtype=torch.float32),
        "duration_chosen": torch.tensor([b.duration_chosen for b in batch], dtype=torch.float32),
        "duration_rejected": torch.tensor([b.duration_rejected for b in batch], dtype=torch.float32),
        "pair_index": torch.tensor([b.pair_index for b in batch], dtype=torch.long),
    }


def pair_meta_from_dataset(
    split: str = "train",
    limit: int | None = None,
    *,
    token: str | None = None,
    skip: int = 0,
) -> dict[int, PairMeta]:
    """Build a {pair_index -> PairMeta} map by streaming the dataset.

    Does NOT decode audio (skips the heavy field). Reuses the same parquet
    iteration order as src/natscore/data/speechjudge.py so pair_index
    semantics match.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from natscore.data.speechjudge import DATASET_ID, _list_split_shards

    shards = _list_split_shards(split, token=token)
    out: dict[int, PairMeta] = {}
    rows_seen = 0
    pairs_yielded = 0
    for shard_name in shards:
        if limit is not None and pairs_yielded >= limit:
            break
        local = hf_hub_download(
            repo_id=DATASET_ID, filename=shard_name,
            repo_type="dataset", token=token,
        )
        parquet = pq.ParquetFile(local)
        cols = ["naturalness_label", "naturalness_annotation", "chosen"]
        # Only ask for the small columns; this saves >100x bandwidth vs reading audio.
        cols = [c for c in cols if c in parquet.schema_arrow.names]
        for rg_idx in range(parquet.num_row_groups):
            if limit is not None and pairs_yielded >= limit:
                break
            table = parquet.read_row_group(rg_idx, columns=cols)
            for row in table.to_pylist():
                if rows_seen < skip:
                    rows_seen += 1
                    continue
                if limit is not None and pairs_yielded >= limit:
                    break
                out[rows_seen] = PairMeta(
                    pair_index=rows_seen,
                    naturalness_label=row.get("naturalness_label", "") or "",
                    naturalness_annotation=row.get("naturalness_annotation") or [],
                    chosen=bool(row.get("chosen", False)),
                )
                rows_seen += 1
                pairs_yielded += 1
    return out
