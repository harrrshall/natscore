"""Unit tests for `natscore.score.Scorer`.

Uses a fake `WhisperFeatureExtractor` stand-in so these tests are fast and
require no Whisper / HF downloads. The real download-and-load path is
covered separately by the network-gated test in `test_scaffold.py`.
"""

from __future__ import annotations

import torch

from natscore.compare import Pair
from natscore.features import WhisperFeatureMeta
from natscore.model import NatScoreHead, NatScoreHeadConfig
from natscore.score import Scorer


class _FakeExtractor:
    """Minimal duck-type of `WhisperFeatureExtractor` for unit tests.

    Returns deterministic feature tensors keyed on the order audios are
    passed in, so the scorer's batch handling is testable without the
    encoder.
    """

    def __init__(self, n_hidden_states: int = 13, hidden_dim: int = 768) -> None:
        self._meta = WhisperFeatureMeta(
            model_name="fake",
            n_layers=n_hidden_states - 1,
            n_hidden_states=n_hidden_states,
            hidden_dim=hidden_dim,
            sample_rate=16_000,
            frame_rate=50,
            max_audio_seconds=30,
            output_frames=1500,
        )
        self._device = torch.device("cpu")

    @property
    def meta(self) -> WhisperFeatureMeta:
        return self._meta

    @property
    def device(self) -> torch.device:
        return self._device

    def batch_extract_layerwise(self, audios):
        # Use a sum() of the input "audio" (a scalar tag in these tests) as
        # the seed so the same input deterministically produces the same
        # features across calls.
        out = []
        for a in audios:
            g = torch.Generator().manual_seed(int(a) if isinstance(a, int) else hash(a) % (2**31))
            t = torch.randn(
                self._meta.n_hidden_states,
                self._meta.output_frames,
                self._meta.hidden_dim,
                generator=g,
            )
            out.append(t)
        return torch.stack(out, dim=0)  # [B, H, T, D]


def _make_scorer() -> Scorer:
    extractor = _FakeExtractor()
    cfg = NatScoreHeadConfig(
        n_hidden_states=extractor.meta.n_hidden_states,
        hidden_dim=extractor.meta.hidden_dim,
        pooler_bottleneck_dim=256,
        score_bottleneck_dim=256,
        dropout=0.0,
    )
    head = NatScoreHead(cfg)
    return Scorer(extractor=extractor, head=head)


def test_score_returns_float() -> None:
    scorer = _make_scorer()
    s = scorer.score(audio=42)
    assert isinstance(s, float)


def test_score_is_deterministic_for_same_input() -> None:
    scorer = _make_scorer()
    a = scorer.score(7)
    b = scorer.score(7)
    assert a == b


def test_batch_score_length_matches_input() -> None:
    scorer = _make_scorer()
    scores = scorer.batch_score([1, 2, 3, 4])
    assert len(scores) == 4
    assert all(isinstance(s, float) for s in scores)


def test_batch_score_empty_returns_empty() -> None:
    scorer = _make_scorer()
    assert scorer.batch_score([]) == []


def test_batch_score_matches_single_score() -> None:
    # Batched and single forwards can drift at ~1e-8 due to non-associative
    # floating-point reductions in attention pooling; require near-equality,
    # not bit-equality.
    import pytest

    scorer = _make_scorer()
    single = [scorer.score(i) for i in (10, 11, 12)]
    batched = scorer.batch_score([10, 11, 12])
    assert single == pytest.approx(batched, abs=1e-5)


def test_compare_returns_pair_with_consistent_winner() -> None:
    scorer = _make_scorer()
    p = scorer.compare(audio_a=1, audio_b=2)
    assert isinstance(p, Pair)
    if p.score_a > p.score_b:
        assert p.winner == "a"
    elif p.score_b > p.score_a:
        assert p.winner == "b"
    else:
        assert p.winner == "tie"


def test_head_runs_in_eval_mode() -> None:
    # Eval mode matters because the trained head has dropout=0.1; running in
    # train mode would inject noise into inference scores.
    scorer = _make_scorer()
    assert not scorer.head.training
