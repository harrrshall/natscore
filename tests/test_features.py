"""Unit tests for the M2 feature pipeline.

Tests fall into two tiers:

1. **Offline tests** (always run in CI): cover the audio decoder, the
   feature cache I/O, and the SpeechJudge ClipRecord dataclass. No
   network, no model download.

2. **Whisper tests** (skipped in CI; run locally with --whisper): load
   the real Whisper-small checkpoint, run forward on a 2-second sine
   wave, assert shapes/determinism. ~150 MB one-time download; ~5s
   forward on CPU. Skipped unless RUN_WHISPER_TESTS=1 is set.
"""

from __future__ import annotations

import io
import os
import wave

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------- audio fixture


def _sine_wav_bytes(seconds: float = 2.0, freq: float = 440.0, sr: int = 16_000) -> bytes:
    """Generate a tiny mono 16-bit PCM WAV of a pure sine."""
    t = np.arange(int(seconds * sr)) / sr
    samples = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ----------------------------------------------------------- feature cache I/O


def test_feature_cache_roundtrip(tmp_path):
    from natscore.data.feature_cache import FeatureCache

    cache = FeatureCache(tmp_path)
    feats = torch.randn(13, 1500, 768, dtype=torch.float16)
    entry = cache.write(
        clip_id="00000000_A",
        hidden_states=feats,
        pair_index=0,
        side="A",
        subset="regular",
        language_setting="en2en",
        sample_rate=24_000,
        duration_seconds=2.0,
        target_text="hello world",
    )
    assert entry.shape == [13, 1500, 768]
    assert entry.dtype == "float16"
    assert "00000000_A" in cache
    loaded = cache.load("00000000_A")
    assert loaded.shape == feats.shape
    assert loaded.dtype == feats.dtype
    assert torch.equal(loaded, feats)


def test_feature_cache_resumes_existing(tmp_path):
    from natscore.data.feature_cache import FeatureCache

    cache_a = FeatureCache(tmp_path)
    cache_a.write(
        clip_id="x", hidden_states=torch.zeros(13, 1500, 768, dtype=torch.float16),
        pair_index=0, side="A", subset="", language_setting="",
        sample_rate=16_000, duration_seconds=0.0, target_text="",
    )

    # A fresh cache instance pointing at the same dir sees the prior write.
    cache_b = FeatureCache(tmp_path)
    assert "x" in cache_b
    assert len(cache_b) == 1


def test_cache_entry_serialization():
    from natscore.data.feature_cache import CacheEntry

    e = CacheEntry(
        clip_id="00000001_B", pair_index=1, side="B", subset="expressive",
        language_setting="zh2zh", sample_rate=24_000, duration_seconds=3.5,
        target_text="测试", file="features/00000001_B.safetensors",
        dtype="float16", shape=[13, 1500, 768],
    )
    s = e.to_json()
    restored = CacheEntry.from_json(s)
    assert restored == e


# ------------------------------------------------------------ clip record dataclass


def test_clip_record_id_format():
    from natscore.data.speechjudge import ClipRecord

    r = ClipRecord(
        pair_index=42, side="A", subset="regular", language_setting="en2en",
        target_text="hi", waveform=np.zeros(16_000, dtype=np.float32), sample_rate=16_000,
    )
    assert r.clip_id == "00000042_A"
    assert r.num_samples() == 16_000
    assert abs(r.duration_seconds() - 1.0) < 1e-6


# --------------------------------------------------------------------- Whisper


# Lives behind an env flag because it downloads ~150 MB the first time and
# takes ~5s on CPU. Worth running locally before any extraction run; not
# worth running in CI on every PR.
_run_whisper = os.environ.get("RUN_WHISPER_TESTS") == "1"


@pytest.mark.skipif(not _run_whisper, reason="set RUN_WHISPER_TESTS=1 to run")
def test_whisper_extract_shape():
    from natscore.features import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor()
    audio = _sine_wav_bytes(seconds=2.0)
    feats = extractor.extract_layerwise(audio)
    meta = extractor.meta

    assert feats.shape == (meta.n_hidden_states, meta.output_frames, meta.hidden_dim)
    assert torch.isfinite(feats).all(), "encoder produced NaN/Inf"


@pytest.mark.skipif(not _run_whisper, reason="set RUN_WHISPER_TESTS=1 to run")
def test_whisper_extract_deterministic():
    from natscore.features import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor()
    audio = _sine_wav_bytes(seconds=1.5)
    a = extractor.extract_layerwise(audio)
    b = extractor.extract_layerwise(audio)
    assert torch.equal(a, b), "encoder is not bitwise-deterministic across calls"


@pytest.mark.skipif(not _run_whisper, reason="set RUN_WHISPER_TESTS=1 to run")
def test_whisper_batch_matches_single():
    from natscore.features import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor()
    a1 = _sine_wav_bytes(seconds=1.0, freq=220.0)
    a2 = _sine_wav_bytes(seconds=1.5, freq=440.0)
    batched = extractor.batch_extract_layerwise([a1, a2])
    assert batched.shape[0] == 2
    # Single-call vs batched must agree (modulo numerical noise).
    single1 = extractor.extract_layerwise(a1)
    assert torch.allclose(batched[0], single1, atol=1e-4)
