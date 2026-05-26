"""M0 scaffold smoke tests — verifies the package installs and imports cleanly.

Replaced/expanded in later milestones (test_model.py, test_score_api.py, etc).
"""

from __future__ import annotations

import subprocess
import sys

import natscore


def test_version_is_string() -> None:
    assert isinstance(natscore.__version__, str)
    assert natscore.__version__


def test_public_symbols_exist() -> None:
    # Package exposes the documented surface even if implementations are stubs.
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


def test_load_raises_not_implemented() -> None:
    # Documents the M0 state honestly; this test flips in M6.
    import pytest

    with pytest.raises(NotImplementedError):
        natscore.load()
