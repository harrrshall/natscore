"""Build the {pair_index -> naturalness_label/annotation/chosen} sidecar.

The feature cache only stores per-clip data; the BT training step needs
per-pair labels too. This script streams the cheap (small) columns of
SpeechJudge-Data and writes a JSON map next to the cache.

Usage:
    python scripts/build_pair_meta.py --limit 30 --output cache/sanity/pair_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from natscore.data.pair_dataset import pair_meta_from_dataset  # noqa: E402


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        cache_token = Path.home() / ".cache" / "huggingface" / "token"
        if cache_token.exists():
            token = cache_token.read_text().strip()
    if not token:
        raise SystemExit("HF_TOKEN not found; run `huggingface-cli login`.")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of pairs to record (None = full split).")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    meta = pair_meta_from_dataset(
        split=args.split, limit=args.limit, token=_require_token(), skip=args.skip,
    )
    serializable = {str(k): asdict(v) for k, v in meta.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(serializable, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(meta)} pair_meta entries -> {args.output}")


if __name__ == "__main__":
    main()
