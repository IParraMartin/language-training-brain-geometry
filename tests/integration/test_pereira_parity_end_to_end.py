"""Real-data end-to-end strict Pereira parity: custom pipeline vs official.

Exports both parity bundles for a pinned GPT-2 revision at one block and runs
the strict comparator, asserting the custom pipeline reproduces the official one.

This is HEAVY (downloads the model + Pereira neural data, runs the official
benchmark) and is therefore OPT-IN: it only runs when BRAIN_CODE_RUN_INTEGRATION=1.
The CPU-only unit suite skips it. `tests/test_parity_metric_shape.py` guards the
underlying assembly-shape fix cheaply and always runs.

Tolerances: the two pipelines are independent fp32 forward passes, so activations
agree only to fp32 accumulation-order noise (~1e-5, Pearson r = 1.0), while the
neural targets are bit-identical and the final Brain-Score agrees to ~6e-7. The
strict defaults (activation_atol=1e-6, score_atol=1e-8) are below that fp32 floor;
these tolerances are set at the cross-implementation fp32 floor while keeping the
neural-target comparison exact (target_atol/rtol stay at their 0.0 defaults).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_RUN = os.environ.get("BRAIN_CODE_RUN_INTEGRATION") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN,
    reason="set BRAIN_CODE_RUN_INTEGRATION=1 to run the real-data Pereira parity "
    "(downloads the model + neural data and runs the official benchmark)"
)

_REPO = Path(__file__).resolve().parents[2]
_MODEL = "openai-community/gpt2"
_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # immutable gpt2 revision
_LAYER = "transformer.h.6"
_BENCH = "Pereira2018.243sentences-ridge"


def _export(side: str, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/export_pereira_parity.py",
            "--side",
            side,
            "--benchmark",
            _BENCH,
            "--model-id",
            _MODEL,
            "--model-revision",
            _REV,
            "--tokenizer-revision",
            _REV,
            "--layer",
            _LAYER,
            "--output",
            str(output)
        ],
        cwd=_REPO,
        check=True
    )


def test_custom_pipeline_reproduces_official_pereira_parity(tmp_path):
    official = tmp_path / "official.npz"
    custom = tmp_path / "custom.npz"
    report = tmp_path / "parity_report.json"

    _export("official", official)
    _export("custom", custom)  # must not raise the single-neuroid-coord stacking error

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pereira_parity.py",
            "--official-bundle",
            str(official),
            "--custom-bundle",
            str(custom),
            "--output",
            str(report),
            # fp32 cross-implementation floor; neural targets stay exact (default 0.0)
            "--activation-atol",
            "1e-4",
            "--activation-rtol",
            "1e-3",
            "--prediction-atol",
            "1e-4",
            "--prediction-rtol",
            "1e-3",
            "--score-atol",
            "1e-5",
            "--score-rtol",
            "1e-4"
        ],
        cwd=_REPO
    )

    assert report.is_file(), "comparator did not write a report"
    parity = json.loads(report.read_text())
    failed = [name for name, check in parity["checks"].items() if not check.get("passed")]
    assert parity["validated_against_official"] is True, f"parity failed on: {failed}"
    # neural targets must match exactly (they are the same data on both sides)
    assert parity["checks"]["neural_targets"]["passed"] is True
    assert result.returncode == 0
