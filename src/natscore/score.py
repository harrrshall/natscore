"""Scorer: end-to-end naturalness scoring API.

Composes the frozen Whisper encoder (`WhisperFeatureExtractor`) with the
trained Bradley-Terry head (`NatScoreHead`) and exposes the three calls
documented in `natscore.__init__`:

    scorer.score(audio)            -> float
    scorer.batch_score(audios)     -> list[float]
    scorer.compare(a, b)           -> Pair

Construction is dependency-injection friendly (pass in an extractor and a
head) so unit tests can stand in fakes without paying for the Whisper
download. End users build a Scorer via `natscore.load(model_id)`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .compare import Pair
from .features import AudioInput, WhisperFeatureExtractor
from .model import NatScoreHead


class Scorer:
    """Stateful naturalness scorer. Holds a frozen Whisper encoder + trained head.

    The Scorer is single-threaded; if you need concurrent scoring, wrap calls
    in your own executor and pin one Scorer per worker (the head is tiny but
    the encoder allocates GPU memory).
    """

    def __init__(
        self,
        extractor: WhisperFeatureExtractor,
        head: NatScoreHead,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = extractor.device
        self._device = torch.device(device)
        self._extractor = extractor
        self._head = head.to(self._device).eval()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def head(self) -> NatScoreHead:
        return self._head

    @property
    def extractor(self) -> WhisperFeatureExtractor:
        return self._extractor

    @torch.inference_mode()
    def score(self, audio: AudioInput) -> float:
        """Score a single audio clip. Higher = more natural."""
        return self.batch_score([audio])[0]

    @torch.inference_mode()
    def batch_score(self, audios: Sequence[AudioInput] | Iterable[AudioInput]) -> list[float]:
        """Score N audio clips in one forward pass."""
        audios = list(audios)
        if len(audios) == 0:
            return []
        feats = self._extractor.batch_extract_layerwise(audios)  # [B, H, T, D]
        # The head was trained in fp32 even when the encoder ran in fp16
        # (the trainer keeps the head in fp32 for stable gradients). Cast
        # features to the head's parameter dtype before forward.
        head_dtype = next(self._head.parameters()).dtype
        logits = self._head(feats.to(head_dtype))                # [B]
        return logits.detach().to("cpu").tolist()

    @torch.inference_mode()
    def compare(self, audio_a: AudioInput, audio_b: AudioInput) -> Pair:
        """Pairwise comparison. Returns a `Pair` with scores, margin, winner, P(A wins)."""
        s_a, s_b = self.batch_score([audio_a, audio_b])
        return Pair.from_scores(s_a, s_b)

    def __repr__(self) -> str:
        return (
            f"Scorer(extractor={self._extractor!r}, "
            f"trainable_params={self._head.trainable_param_count()}, "
            f"device={self._device})"
        )
