"""Rebuilt statistics reproduce the reported statistics; skipped when the results bundle is absent."""

from __future__ import annotations

import pytest

from brain_code.reproduction.config import load_paths

bundle_root = load_paths().results_bundle
if not (bundle_root / "MANIFEST.json").exists():
    pytest.skip("results bundle not built", allow_module_level=True)

from brain_code.reproduction.analysis import (
    adaptation_results,
    geometry_results,
    layer_results,
    primary_results
)
from brain_code.reproduction.bundle import Bundle

# fixed-random-reference family at the precision it is reported with
FIXED_REFERENCE = {
    ("pereira_243", "L-R"): (0.057, 2.18e-5),
    ("pereira_243", "C-R"): (0.004, 0.0586),
    ("pereira_243", "L-C"): (0.053, 1.89e-8),
    ("tuckute", "L-R"): (0.051, 8.50e-4),
    ("tuckute", "C-R"): (-0.007, 0.0211),
    ("tuckute", "L-C"): (0.059, 7.51e-5),
    ("blank", "L-R"): (0.014, 3.67e-4),
    ("blank", "C-R"): (0.009, 1.80e-6),
    ("blank", "L-C"): (0.005, 0.0134),
    ("fedorenko", "L-R"): (-0.004, 0.0215),
    ("fedorenko", "C-R"): (-0.005, 2.59e-4),
    ("fedorenko", "L-C"): (0.000, 0.809)
}


@pytest.fixture(scope="module")
def bundle():
    return Bundle(bundle_root)


def test_fixed_reference_family(bundle):
    rows = primary_results(bundle)["family"]["fixed_reference"]
    for row in rows:
        estimate, p = FIXED_REFERENCE[(row["benchmark"], row["contrast"])]
        assert round(row["estimate"], 3) == pytest.approx(estimate, abs=6e-4)
        assert row["p"] == pytest.approx(p, rel=0.02)


def test_sampled_reference_family_keeps_the_language_code_contrasts(bundle):
    rows = {(r["benchmark"], r["contrast"]): r for r in primary_results(bundle)["family"]["sampled_reference"]}
    assert rows[("pereira_243", "L-C")]["significant"] and rows[("tuckute", "L-C")]["significant"]
    assert not any(rows[(b, "L-R")]["significant"] for b in (
        "pereira_243",
        "tuckute",
        "blank",
        "fedorenko"
    ))


def test_depth_profile_counts(bundle):
    layers = layer_results(bundle)
    assert layers["pereira_243"]["gap_significant_blocks"] == 11
    assert layers["pereira_384"]["gap_significant_blocks"] == 11
    assert layers["pereira_243"]["heterogeneity_significant_blocks"] == [0, 2]
    assert layers["pereira_243"]["exact_order_blocks"] == 0


def test_geometry_summaries(bundle):
    geo = geometry_results(bundle)
    assert round(geo["seed_cka_blocks_4_11"]["tinystories"], 3) == 0.899
    assert round(geo["language_minus_language_code_cka"][4], 3) == 0.394
    assert geo["n_models"]["random"] == 2


def test_adaptation_peak_test(bundle):
    peak = adaptation_results(bundle)["paired_peak"]
    assert peak["t"] == pytest.approx(4.06, abs=0.01)
    assert peak["p"] == pytest.approx(0.027, abs=0.001)
