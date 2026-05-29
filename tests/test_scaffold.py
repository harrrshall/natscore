"""Package smoke tests.

`test_load_*` originally documented the M0 NotImplementedError; M6 flips it
to verify the real `natscore.load()` path. The end-to-end test is
network-gated (RUN_HUB_TESTS=1) because it downloads Whisper-small
(~290 MB) and the NatScore checkpoint from HF Hub.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import natscore


def test_version_is_string() -> None:
    assert isinstance(natscore.__version__, str)
    assert natscore.__version__


def test_public_symbols_exist() -> None:
    for name in ("load", "Scorer", "Pair"):
        assert hasattr(natscore, name), f"natscore.{name} missing from public API"


def test_cli_help_works() -> None:
    # `natscore --help` must succeed; this is the M0 exit criterion.
    result = subprocess.run(
        [sys.executable, "-m", "natscore.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "NatScore" in result.stdout


def test_short_alias_resolves() -> None:
    from natscore import _SHORT_ALIASES, DEFAULT_MODEL_ID

    assert _SHORT_ALIASES["natscore-small-v0"] == DEFAULT_MODEL_ID


_run_hub = os.environ.get("RUN_HUB_TESTS") == "1"


@pytest.mark.skipif(not _run_hub, reason="set RUN_HUB_TESTS=1 to run (downloads ~290 MB)")
def test_load_returns_scorer_with_trained_head() -> None:
    scorer = natscore.load()
    assert isinstance(scorer, natscore.Scorer)
    # The released head is 394,255 params (see model card). Allow drift if
    # future checkpoints retune the bottleneck dims; just enforce non-trivial.
    n = scorer.head.trainable_param_count()
    assert n > 100_000, f"head looks empty: {n} params"

    # A 2-second sine wave should produce a scalar logit without error.
    import io
    import wave
    import numpy as np
    t = np.arange(int(2.0 * 16_000)) / 16_000
    samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(samples.tobytes())

    s = scorer.score(buf.getvalue())
    assert isinstance(s, float)
    assert -50.0 < s < 50.0  # sanity range for a logit

    # batch_score and compare exercise the same forward path.
    pair = scorer.compare(buf.getvalue(), buf.getvalue())
    assert pair.score_a == pytest.approx(pair.score_b, abs=1e-4)
    assert pair.winner == "tie" or abs(pair.margin) < 1e-3
