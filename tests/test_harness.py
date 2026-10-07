"""Tests for benchmark-score ceiling handling.

Checks that:
  - The Pereira-243 ceiling (0.32) is applied only to Pereira-243.
  - All other benchmarks return paper_normalized=None.
  - BENCHMARK_METADATA contains the required keys for every registered benchmark.
  - resolve_smoke_layer resolves 'first', 'middle', 'last', and integer specs.
"""

from __future__ import annotations

import pytest

# importable without brainscore_language, which the harness imports lazily
from brain_code.experiments.brain_score_harness import _compute_paper_normalized, get_benchmark_meta
from brain_code.config import EXPERIMENTS


# every benchmark ID registered in config must appear in BENCHMARK_METADATA
REQUIRED_BENCHMARK_IDS = list(EXPERIMENTS.values())

REQUIRED_META_KEYS = {
    "human_name",
    "modality",
    "neural_target",
    "stimulus_granularity",
    "raw_metric",
    "paper_ceiling",
    "paper_ceiling_source",
    "paper_ceiling_valid",
    "notes"
}


@pytest.mark.parametrize("benchmark_id", REQUIRED_BENCHMARK_IDS)
def test_benchmark_metadata_has_required_keys(benchmark_id):
    meta = get_benchmark_meta(benchmark_id)
    missing = REQUIRED_META_KEYS - set(meta.keys())
    assert not missing, (
        f"BENCHMARK_METADATA[{benchmark_id!r}] is missing keys: {missing}"
    )


def test_pereira_243_ceiling_applied():
    """Pereira-243 must produce a non-None paper_normalized."""
    raw = 0.20
    paper_norm, reason = _compute_paper_normalized(raw, "Pereira2018.243sentences-ridge")
    assert paper_norm is not None, "Pereira-243 should have a valid paper_normalized."
    assert abs(paper_norm - raw / 0.32) < 1e-9, (
        f"Expected paper_normalized = {raw / 0.32:.6f}, got {paper_norm:.6f}"
    )


@pytest.mark.parametrize("benchmark_id", [
    "Pereira2018.384sentences-ridge",
    "Blank2014-ridge",
    "Fedorenko2016-ridge",
    "Tuckute2024-ridge",
    "Futrell2018-pearsonr"
])
def test_ceiling_not_applied_to_other_benchmarks(benchmark_id):
    """No other registered benchmark should produce a non-None paper_normalized."""
    paper_norm, reason = _compute_paper_normalized(0.20, benchmark_id)
    assert paper_norm is None, (
        f"{benchmark_id!r}: paper_normalized should be None but got {paper_norm!r}.  "
        f"Reason provided: {reason}"
    )
    assert reason, f"{benchmark_id!r}: a reason string must be provided when paper_normalized is None."


def test_fedorenko_labeled_as_ecog():
    """Fedorenko2016 must be labelled as ECoG, not fMRI."""
    meta = get_benchmark_meta("Fedorenko2016-ridge")
    assert meta["modality"] == "ECoG", (
        "Fedorenko2016 is ECoG data.  Do not describe it as fMRI."
    )


def test_futrell_labeled_as_behavioral():
    """Futrell2018 must be labelled as behavioral."""
    meta = get_benchmark_meta("Futrell2018-pearsonr")
    assert meta["modality"] == "behavioral", (
        "Futrell2018 is a behavioral benchmark, not a neural one."
    )


def test_unknown_benchmark_returns_safe_defaults():
    """Unregistered benchmark IDs must return paper_normalized=None with a reason."""
    paper_norm, reason = _compute_paper_normalized(0.5, "SomeFutureBenchmark2030")
    assert paper_norm is None
    assert reason


def test_resolve_smoke_layer_first_last_middle():
    """resolve_smoke_layer correctly maps named specs."""
    from brain_code.experiments.brain_score_harness import CustomModelHarness

    harness = object.__new__(CustomModelHarness)
    harness.layer_names = [f"transformer.h.{i}" for i in range(12)]

    assert harness.resolve_smoke_layer("first") == "transformer.h.0"
    assert harness.resolve_smoke_layer("last") == "transformer.h.11"
    assert harness.resolve_smoke_layer("middle") == "transformer.h.6"
    assert harness.resolve_smoke_layer("5") == "transformer.h.5"


def test_resolve_smoke_layer_integer_out_of_range():
    from brain_code.experiments.brain_score_harness import CustomModelHarness

    harness = object.__new__(CustomModelHarness)
    harness.layer_names = [f"transformer.h.{i}" for i in range(4)]

    with pytest.raises(ValueError, match="out of range"):
        harness.resolve_smoke_layer("10")


def test_resolve_smoke_layer_invalid_spec():
    from brain_code.experiments.brain_score_harness import CustomModelHarness

    harness = object.__new__(CustomModelHarness)
    harness.layer_names = [f"transformer.h.{i}" for i in range(4)]

    with pytest.raises(ValueError, match="must be"):
        harness.resolve_smoke_layer("badspec")


def test_run_experiment_rejects_invalid_layer_selection_mode():
    """run_experiment must raise ValueError for any unknown layer_selection_mode."""
    from brain_code.experiments.brain_score_harness import run_experiment
    with pytest.raises(ValueError, match="layer_selection_mode"):
        run_experiment(
            hf_model_id="gpt2",
            benchmark_identifier="Pereira2018.243sentences-ridge",
            layer_selection_mode="argmax_on_test_set"  # forbidden
        )


def test_run_experiment_accepts_valid_layer_selection_modes(monkeypatch):
    """Valid layer_selection_mode values must pass the layer-selection guard.

    What happens *after* that guard is environment-dependent: with
    ``brainscore_language`` absent the harness raises a "brainscore_language"
    RuntimeError; with it installed the harness proceeds to resolve the model.
    We stub model resolution so the test never touches HuggingFace (unit tests
    are CPU-only, no downloads) and assert only the invariant this test owns:
    a valid mode does NOT raise the layer_selection_mode ValueError.
    """
    from brain_code.experiments import brain_score_harness

    def _no_network(*_args, **_kwargs):
        raise RuntimeError("model resolution reached (stubbed; no network)")

    monkeypatch.setattr(brain_score_harness, "_resolve_model_source", _no_network)

    for mode in ["descriptive_grid", "prespecified"]:
        # either the brainscore-missing RuntimeError or the stubbed resolver
        # proves the mode guard passed; neither is a mode ValueError
        with pytest.raises(RuntimeError) as exc_info:
            brain_score_harness.run_experiment(
                hf_model_id="gpt2",
                benchmark_identifier="Pereira2018.243sentences-ridge",
                layer_selection_mode=mode,
                model_revision="a" * 40,
                prespecified_layer="last"  # needed when mode="prespecified"
            )
        assert "layer_selection_mode" not in str(exc_info.value)


def test_compute_paper_normalized_returns_legacy_key_name():
    """The harness must compute the legacy_schrimpf2021_pereira243_normalized value correctly."""
    raw = 0.16
    paper_norm, reason = _compute_paper_normalized(raw, "Pereira2018.243sentences-ridge")
    assert paper_norm is not None
    assert abs(paper_norm - 0.5) < 1e-9, f"Expected 0.5, got {paper_norm}"


def test_pereira384_ceiling_is_null():
    """Pereira-384 must return None for legacy_schrimpf2021_pereira243_normalized."""
    paper_norm, reason = _compute_paper_normalized(0.16, "Pereira2018.384sentences-ridge")
    assert paper_norm is None, (
        "Pereira-384 must have null legacy ceiling — 0.32 only applies to Pereira-243"
    )
    assert "384" in reason or "ceiling" in reason.lower()
