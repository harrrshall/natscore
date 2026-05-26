"""NatScore — preference-supervised naturalness scorer for modern neural TTS.

Public API (stable surface; implementations land in Milestones 2-6):

    natscore.load(model_id)        -> Scorer
    scorer.score(audio)            -> float
    scorer.batch_score(paths)      -> list[float]
    scorer.compare(a, b)           -> Pair

See PROJECT_PLAN.md §2.1 for the full specification.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__", "load", "Scorer", "Pair"]


def load(model_id: str = "natscore-small-v1"):
    """Load a scorer checkpoint from HuggingFace Hub.

    Not implemented in Milestone 0 — the trained checkpoint does not exist yet.
    Lands in Milestone 6 once a checkpoint is uploaded to HF Hub.
    """
    raise NotImplementedError(
        "natscore.load() is not implemented in the M0 scaffold. "
        "Trained checkpoints land in Milestone 6 — see PROJECT_PLAN.md §8."
    )


class Scorer:
    """Placeholder. Real implementation lands in src/natscore/score.py (M3)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Scorer is not implemented in the M0 scaffold. See PROJECT_PLAN.md §8."
        )


class Pair:
    """Placeholder. Real dataclass lands in src/natscore/compare.py (M3)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Pair is not implemented in the M0 scaffold. See PROJECT_PLAN.md §8."
        )
