"""On-disk cache of Whisper encoder features.

Layout:

    cache_dir/
        manifest.jsonl              # one JSON object per cached clip
        features/
            00000000_A.safetensors  # one file per clip
            00000000_B.safetensors
            00000001_A.safetensors
            ...

Each .safetensors file holds a single tensor under key "hidden_states" with
shape (n_hidden_states, output_frames, hidden_dim) in the dtype it was
written with (typically float16 to halve disk).

The manifest is append-only JSONL so resume-from-checkpoint and concurrent
writers (split across multiple Kaggle/Modal jobs) are trivial. On read we
just iterate the JSONL and dedupe by `clip_id` keeping the last entry.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file, save_file

MANIFEST_NAME = "manifest.jsonl"
FEATURES_SUBDIR = "features"
TENSOR_KEY = "hidden_states"


@dataclass(frozen=True)
class CacheEntry:
    clip_id: str
    pair_index: int
    side: str                       # "A" | "B"
    subset: str
    language_setting: str
    sample_rate: int
    duration_seconds: float
    target_text: str
    file: str                       # relative path under cache_dir
    dtype: str                      # e.g. "float16"
    shape: list[int]                # [H, T, D]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> CacheEntry:
        return cls(**json.loads(line))


class FeatureCache:
    """File-based cache. One instance per cache_dir.

    Thread-safe for append; not safe for concurrent processes writing the
    SAME clip_id (we just assume the orchestrator partitions by index).
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / FEATURES_SUBDIR).mkdir(exist_ok=True)
        self._manifest_path = self.dir / MANIFEST_NAME
        self._lock = threading.Lock()
        self._existing: set[str] | None = None

    # --------------------------------------------------------------- existence

    def existing_clip_ids(self) -> set[str]:
        """Set of clip_ids already in the manifest. Cached after first call."""
        if self._existing is None:
            ids: set[str] = set()
            if self._manifest_path.exists():
                with self._manifest_path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ids.add(json.loads(line)["clip_id"])
                        except (json.JSONDecodeError, KeyError):
                            continue
            self._existing = ids
        return self._existing

    def __contains__(self, clip_id: str) -> bool:
        return clip_id in self.existing_clip_ids()

    # --------------------------------------------------------------------- io

    def _file_for(self, clip_id: str) -> Path:
        return self.dir / FEATURES_SUBDIR / f"{clip_id}.safetensors"

    def write(
        self,
        clip_id: str,
        hidden_states: torch.Tensor,
        *,
        pair_index: int,
        side: str,
        subset: str,
        language_setting: str,
        sample_rate: int,
        duration_seconds: float,
        target_text: str,
    ) -> CacheEntry:
        """Persist one clip's features and append a manifest entry."""
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must be [H, T, D]; got shape {tuple(hidden_states.shape)}"
            )
        path = self._file_for(clip_id)
        # safetensors requires contiguous CPU tensors.
        tensor = hidden_states.detach().contiguous().to("cpu")
        save_file({TENSOR_KEY: tensor}, str(path))

        entry = CacheEntry(
            clip_id=clip_id,
            pair_index=pair_index,
            side=side,
            subset=subset,
            language_setting=language_setting,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            target_text=target_text,
            file=str(path.relative_to(self.dir)),
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=list(tensor.shape),
        )
        with self._lock:
            with self._manifest_path.open("a") as fh:
                fh.write(entry.to_json() + "\n")
            if self._existing is not None:
                self._existing.add(clip_id)
        return entry

    def load(self, clip_id: str) -> torch.Tensor:
        """Load one clip's features back into memory."""
        path = self._file_for(clip_id)
        if not path.exists():
            raise FileNotFoundError(f"No cached features for clip_id={clip_id!r} at {path}")
        return load_file(str(path))[TENSOR_KEY]

    def iter_manifest(self) -> Iterator[CacheEntry]:
        """Yield every manifest entry in write order (with duplicates if any)."""
        if not self._manifest_path.exists():
            return
        with self._manifest_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield CacheEntry.from_json(line)

    def total_size_bytes(self) -> int:
        return sum(p.stat().st_size for p in (self.dir / FEATURES_SUBDIR).glob("*.safetensors"))

    def __len__(self) -> int:
        return len(self.existing_clip_ids())
