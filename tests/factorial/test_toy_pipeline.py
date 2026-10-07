"""Toy end-to-end pipeline test for the factorial corpora."""

from __future__ import annotations

import sys
import subprocess


def test_build_corpora_toy_mode_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_factorial_corpora",
            "--toy_mode"
        ],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, (
        f"brain-build-corpora --toy_mode exited {result.returncode}.\n"
        f"stderr:\n{result.stderr}"
    )
    assert "TOY MODE COMPLETE" in result.stdout
