"""NatScore: preference-supervised naturalness scorer for modern neural TTS.

Public API:

    natscore.load(model_id)        -> Scorer
    scorer.score(audio)            -> float
    scorer.batch_score(audios)     -> list[float]
    scorer.compare(a, b)           -> Pair

Default release (M5b / M6) lives at `harrrshall/natscore-small-v0` on HF Hub.
Weights are CC-BY-NC-4.0 (inherited from SpeechJudge-Data); code is Apache-2.0.
"""

from __future__ import annotations

from typing import Any

import torch

from .compare import Pair
from .score import Scorer

__version__ = "0.1.0.dev0"

__all__ = ["__version__", "load", "Scorer", "Pair"]

DEFAULT_MODEL_ID = "harrrshall/natscore-small-v0"
DEFAULT_CHECKPOINT_FILENAME = "final.pt"

# Friendly short aliases that map to full HF repo IDs. Lets callers write
# `natscore.load("natscore-small-v0")` without remembering the namespace.
_SHORT_ALIASES = {
    "natscore-small-v0": "harrrshall/natscore-small-v0",
}


def load(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    cache_dir: str | None = None,
    revision: str | None = None,
    encoder_model_name: str = "openai/whisper-small",
) -> Scorer:
    """Load a NatScore checkpoint from HuggingFace Hub and return a ready Scorer.

    Args:
        model_id: HF repo (e.g. `"harrrshall/natscore-small-v0"`) or a short
            alias (e.g. `"natscore-small-v0"`).
        checkpoint_filename: file inside the repo. Defaults to `"final.pt"`.
        device: where to place the scorer. Auto-detects CUDA if available,
            otherwise CPU.
        dtype: encoder dtype. fp32 by default for portability; pass
            `torch.float16` on a GPU to roughly halve encoder memory and
            runtime with negligible score drift.
        cache_dir: passed to `huggingface_hub` for the local cache location.
        revision: optional git revision / tag in the HF repo.
        encoder_model_name: which Whisper backbone to use. Must match the
            checkpoint's training-time encoder; the default is correct for
            every checkpoint currently on the Hub.

    Returns:
        A `Scorer` ready to call `.score()`, `.batch_score()`, `.compare()`.

    Notes:
        First call downloads ~290 MB of Whisper-small weights plus a ~5 MB
        NatScore checkpoint; subsequent calls hit the local HF cache.
    """
    # Lazy-import: `huggingface_hub` is a runtime dep but we only need it here.
    from huggingface_hub import hf_hub_download

    repo_id = _SHORT_ALIASES.get(model_id, model_id)

    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename,
        revision=revision,
        cache_dir=cache_dir,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = _build_head_from_checkpoint(ckpt)
    head = head.to(device).eval()

    # Lazy-import to keep the package import cheap (features.py pulls in
    # transformers + torch + soundfile).
    from .features import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor(
        model_name=encoder_model_name,
        device=device,
        dtype=dtype,
    )

    return Scorer(extractor=extractor, head=head, device=device)


def _build_head_from_checkpoint(ckpt: dict[str, Any]):
    """Reconstruct a `NatScoreHead` from a saved checkpoint dict.

    Reads the head config from `ckpt["config"]["model"]` and strips the
    `module.` prefix from `model_state` keys (added by DataParallel during
    multi-GPU training). The trainer always saves the *unwrapped* state, but
    older checkpoints from earlier in M5b may still carry the prefix; the
    strip is a no-op on cleanly-saved checkpoints.
    """
    from .model import NatScoreHead, NatScoreHeadConfig

    if "model_state" not in ckpt:
        raise KeyError(
            "Checkpoint is missing 'model_state'. Got keys: "
            f"{sorted(ckpt.keys())}. Is this a NatScore checkpoint?"
        )

    model_cfg_raw = ckpt.get("config", {}).get("model", {})
    if not model_cfg_raw:
        raise KeyError(
            "Checkpoint is missing config['model']. Cannot reconstruct head."
        )

    cfg = NatScoreHeadConfig(**model_cfg_raw)
    head = NatScoreHead(cfg)

    state = ckpt["model_state"]
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    head.load_state_dict(state)
    return head
