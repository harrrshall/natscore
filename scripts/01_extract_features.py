"""Milestone 2: extract frozen Whisper encoder features for SpeechJudge clips.

What it does:

1. Stream SpeechJudge-Data clips from HF (decodes inline-bytes audio).
2. Run the frozen Whisper-small encoder forward.
3. Save the per-layer hidden states (shape [H, T, D]) to disk as one
   safetensors file per clip, indexed by an append-only JSONL manifest.

Two non-obvious things:

- **Disk-budget reality check.** PROJECT_PLAN.md s8 estimated 30-50 GB.
  At full per-clip layer-wise resolution with float16 the math actually
  works out to ~5.5 TB (200K clips * 12+1 layers * 1500 frames * 768 dims
  * 2 bytes). Use `--estimate-only` to print the projected size BEFORE
  committing to a multi-hour run. The user must then choose between
  (a) caching a reduced subset, (b) caching pooled features only,
  (c) online extraction during training (no cache).

- **Resume by skipping cached clip_ids.** The manifest doubles as a
  checkpoint; the script reads it on startup and skips any pair_index
  whose A and B sides are both present. Safe to re-run after a crash.

Usage:

    # Smoke run on 100 pairs (200 clips). Writes to ./cache/smoke/
    python scripts/01_extract_features.py --limit 100 --output cache/smoke

    # Print projected disk usage without extracting anything
    python scripts/01_extract_features.py --estimate-only

    # Full extraction (DO NOT run before sanity-checking the estimate first)
    python scripts/01_extract_features.py --output cache/whisper_small_full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from natscore.data.feature_cache import FeatureCache  # noqa: E402
from natscore.data.speechjudge import iter_clips  # noqa: E402
from natscore.features import WhisperFeatureExtractor  # noqa: E402

DEFAULT_OUTPUT = "cache/whisper_small"
SPLIT_PAIR_COUNTS = {
    # Approximate counts from the SpeechJudge-Data card (verify in M1 if needed).
    "train": 42_000,
    "dev": 6_000,
    "test": 50_000,
}


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024
    return f"{f:.2f} PB"


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        # Fall back to the cached token file populated by `huggingface-cli login`.
        cache_token = Path.home() / ".cache" / "huggingface" / "token"
        if cache_token.exists():
            token = cache_token.read_text().strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN not found. Export it or run `huggingface-cli login` first."
        )
    return token


def _estimate(model_name: str, splits: list[str], dtype: torch.dtype) -> None:
    """Print projected disk usage based on encoder geometry alone."""
    extractor = WhisperFeatureExtractor(model_name=model_name, dtype=dtype)
    meta = extractor.meta
    per_clip = meta.bytes_per_clip(dtype=dtype)
    print(f"Model: {model_name}")
    print(f"  Layers (incl. embedding): {meta.n_hidden_states}")
    print(f"  Output frames per clip:   {meta.output_frames}")
    print(f"  Hidden dim:               {meta.hidden_dim}")
    print(f"  Cache dtype:              {dtype}")
    print(f"  Per-clip on-disk size:    {_human_bytes(per_clip)}")
    print()
    grand_total = 0
    for split in splits:
        pairs = SPLIT_PAIR_COUNTS.get(split, 0)
        clips = pairs * 2
        total = clips * per_clip
        grand_total += total
        print(f"  {split:>5s}: ~{pairs:>6d} pairs -> ~{clips:>6d} clips -> {_human_bytes(total)}")
    print(f"  TOTAL: {_human_bytes(grand_total)}")
    print()
    print("If this exceeds your disk budget, options:")
    print("  - Cache a subset only (e.g. dev + a 5K train sample for iteration)")
    print("  - Switch to online extraction during training (no cache)")
    print("  - Cache mean-pooled per-layer embeddings (shape [H, D] only) -> ~750x smaller")


def _extract_split(
    extractor: WhisperFeatureExtractor,
    cache: FeatureCache,
    split: str,
    limit: int | None,
    skip: int,
    token: str,
    cache_dtype: torch.dtype,
) -> None:
    """Stream-extract clips for one split. Resumable via the manifest."""
    existing = cache.existing_clip_ids()
    yielded_pairs = 0
    total_target = limit if limit is not None else SPLIT_PAIR_COUNTS.get(split)
    bar = tqdm(total=total_target, desc=f"{split} pairs", unit="pair")
    t0 = time.time()
    n_extracted = 0

    pair_emitted: dict[int, set[str]] = {}
    for clip in iter_clips(split=split, limit=limit, token=token, skip=skip):
        clip_id = clip.clip_id

        if clip_id in existing:
            pair_emitted.setdefault(clip.pair_index, set()).add(clip.side)
            if pair_emitted[clip.pair_index] >= {"A", "B"}:
                yielded_pairs += 1
                bar.update(1)
            continue

        feats = extractor.extract_layerwise(clip.waveform)  # [H, T, D]
        feats = feats.to(cache_dtype)
        cache.write(
            clip_id,
            feats,
            pair_index=clip.pair_index,
            side=clip.side,
            subset=clip.subset,
            language_setting=clip.language_setting,
            sample_rate=clip.sample_rate,
            duration_seconds=clip.duration_seconds(),
            target_text=clip.target_text,
        )
        n_extracted += 1
        pair_emitted.setdefault(clip.pair_index, set()).add(clip.side)
        if pair_emitted[clip.pair_index] >= {"A", "B"}:
            yielded_pairs += 1
            bar.update(1)

    bar.close()
    dt = time.time() - t0
    if n_extracted:
        size = cache.total_size_bytes()
        mean = size / max(1, len(cache.existing_clip_ids()))
        print(
            f"\n[{split}] extracted {n_extracted} new clips in {dt:.1f}s "
            f"({dt / max(1, n_extracted):.2f} s/clip CPU). "
            f"Cache: {_human_bytes(size)} total, ~{_human_bytes(int(mean))} per clip."
        )
    else:
        print(f"\n[{split}] nothing to extract (manifest already had all target clips).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Cache directory.")
    parser.add_argument("--split", default="train",
                        choices=list(SPLIT_PAIR_COUNTS), help="Dataset split to extract.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max pairs to extract from this split (None = full split).")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first N pairs (for resume / sharding).")
    parser.add_argument("--model", default="openai/whisper-small",
                        help="Whisper checkpoint to load.")
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / cuda:0. Default: cpu (will warn if cuda is available).")
    parser.add_argument("--cache-dtype", default="float16", choices=["float16", "float32"],
                        help="On-disk feature dtype.")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Print projected disk usage and exit without downloading anything.")
    args = parser.parse_args()

    cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32

    if args.estimate_only:
        _estimate(args.model, list(SPLIT_PAIR_COUNTS), cache_dtype)
        return

    token = _require_token()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.cuda.is_available():
        print("[note] CUDA is available but --device cpu was selected.")
    print(f"Loading {args.model} on {device} ...")
    extractor = WhisperFeatureExtractor(model_name=args.model, device=device, dtype=torch.float32)
    print(f"  {extractor!r}")
    print(f"  Per-clip on-disk size: {_human_bytes(extractor.meta.bytes_per_clip(cache_dtype))}")

    cache = FeatureCache(args.output)
    print(f"Cache dir: {cache.dir}  (already has {len(cache)} clips)")

    _extract_split(
        extractor=extractor,
        cache=cache,
        split=args.split,
        limit=args.limit,
        skip=args.skip,
        token=token,
        cache_dtype=cache_dtype,
    )

    final_size = cache.total_size_bytes()
    n = len(cache)
    print(f"\nDone. {n} clips cached, total {_human_bytes(final_size)} on disk.")
    if n:
        print(f"Mean per clip: {_human_bytes(final_size // n)}")
        full_est = (final_size // n) * sum(c * 2 for c in SPLIT_PAIR_COUNTS.values())
        print(f"Extrapolated full-dataset (train+dev+test) size: ~{_human_bytes(full_est)}")


if __name__ == "__main__":
    main()
