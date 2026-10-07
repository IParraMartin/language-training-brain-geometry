"""Tests for the shared full-run data loader (CPU, no brainscore needed).

Covers:
- layer-spec resolution and HuggingFace hidden-state extraction/pooling
- load_neural_benchmark fail-closed policy (via mock xarray assemblies)

The brainscore assembly reader (``load_neural_benchmark``) needs the submodule +
BrainIO cache for a real run; here we use minimal mock assemblies to verify the
fail-closed RuntimeError paths without the submodule.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from brain_code.experiments.data_loading import extract_hidden_states, load_neural_benchmark, resolve_layer_index


class _FakeCoord:
    def __init__(self, vals):
        self.values = vals
        self.dims = ("presentation",)


class _FakeAssembly:
    """Minimal xarray-like assembly stub for fail-closed tests."""

    def __init__(
        self,
        n_stim=10,
        n_neuroid=5,
        ndim=2,
        stim_coord="stimulus",
        group_coord="story",  # matches Pereira2018 official split_coord
        context_coord="passage_label",
        include_stim=True,
        include_group=True,
        include_context=True,
        duplicate_stim_cross_group=False
    ):
        stim_vals = [f"s{i}" for i in range(n_stim)]
        if duplicate_stim_cross_group:
            group_vals = [0 if i < n_stim // 2 else 1 for i in range(n_stim)]
            stim_vals[n_stim // 2] = "s0"
        else:
            group_vals = [i // max(1, n_stim // 5) for i in range(n_stim)]

        self.dims = ("presentation", "neuroid")
        self.coords = {}
        if include_stim:
            self.coords[stim_coord] = _FakeCoord(np.array(stim_vals))
        if include_group:
            self.coords[group_coord] = _FakeCoord(np.array(group_vals))
        if include_context:
            self.coords[context_coord] = _FakeCoord(
                np.array([f"passage-{i // max(1, n_stim // 5)}" for i in range(n_stim)])
            )

        shape = (n_stim, n_neuroid) if ndim == 2 else (n_stim, n_neuroid, 3)
        self.values = np.random.default_rng(0).standard_normal(shape)

    def __getitem__(self, key):
        return self.coords[key]

    def transpose(self, *dims):
        return self


class _FakeBrainscoreModule:
    def __init__(self, assembly):
        self._assembly = assembly

    def load_benchmark(self, bid):
        bm = MagicMock()
        bm.data = self._assembly
        return bm


@contextmanager
def _patch_load_benchmark(assembly):
    """Patch brainscore_language with a fake that returns the given assembly."""
    fake_module = _FakeBrainscoreModule(assembly)
    with patch.dict("sys.modules", {"brainscore_language": fake_module}):
        yield


def test_load_neural_benchmark_raises_on_missing_stimulus_coord():
    """Missing stimulus coordinate must raise RuntimeError (fail-closed)."""
    asm = _FakeAssembly(include_stim=False, include_group=True)
    with _patch_load_benchmark(asm):
        with pytest.raises(RuntimeError, match="stimulus coordinate"):
            load_neural_benchmark("Pereira2018.243sentences-ridge")


def test_load_neural_benchmark_raises_on_missing_group_coord():
    """Missing grouping coordinate must raise RuntimeError — row-fallback is FORBIDDEN."""
    asm = _FakeAssembly(include_stim=True, include_group=False)
    with _patch_load_benchmark(asm):
        with pytest.raises(RuntimeError, match="grouping coordinate"):
            load_neural_benchmark("Pereira2018.243sentences-ridge")


def test_load_neural_benchmark_raises_on_ndim3():
    """Response array with ndim > 2 must raise RuntimeError."""
    asm = _FakeAssembly(ndim=3)
    with _patch_load_benchmark(asm):
        with pytest.raises(RuntimeError, match="ndim=3"):
            load_neural_benchmark("Pereira2018.243sentences-ridge")


def test_load_neural_benchmark_raises_on_cross_group_duplicate_stim():
    """Duplicate stimulus IDs crossing group boundaries must raise RuntimeError."""
    asm = _FakeAssembly(duplicate_stim_cross_group=True, n_stim=10)
    with _patch_load_benchmark(asm):
        with pytest.raises(RuntimeError, match="more than one group"):
            load_neural_benchmark("Pereira2018.243sentences-ridge")


def test_load_neural_benchmark_raises_on_import_failure():
    """Absent brainscore_language must raise RuntimeError with install hint."""
    with patch.dict("sys.modules", {"brainscore_language": None}):
        with pytest.raises((RuntimeError, ImportError)):
            load_neural_benchmark("Pereira2018.243sentences-ridge")


def test_resolve_layer_index_named_and_int():
    assert resolve_layer_index(12, "first") == 0
    assert resolve_layer_index(12, "last") == 11
    assert resolve_layer_index(12, "middle") == 6
    assert resolve_layer_index(4, "2") == 2
    assert resolve_layer_index(4, 3) == 3


@pytest.mark.parametrize("bad", ["99", "-1", "nonsense"])
def test_resolve_layer_index_rejects_bad(bad):
    with pytest.raises(ValueError):
        resolve_layer_index(4, bad)


@pytest.mark.parametrize("pooling", ["last", "mean", "sum"])
def test_extract_hidden_states_shape(micro_model, tiny_tokenizer, pooling):
    sents = ["a b c", "d e f g", "h i"]
    reps = extract_hidden_states(
        micro_model,
        tiny_tokenizer,
        sents,
        layer="last",
        pooling=pooling,
        batch_size=2
    )
    assert reps.shape == (3, micro_model.config.n_embd)
    assert reps.dtype == np.float32
    assert np.isfinite(reps).all()


def test_extract_hidden_states_layers_differ(micro_model, tiny_tokenizer):
    sents = ["a b c", "d e f"]
    r_first = extract_hidden_states(
        micro_model,
        tiny_tokenizer,
        sents,
        layer="first",
        pooling="mean"
    )
    r_last = extract_hidden_states(
        micro_model,
        tiny_tokenizer,
        sents,
        layer="last",
        pooling="mean"
    )
    assert not np.allclose(r_first, r_last)


def test_extract_hidden_states_bad_pooling(micro_model, tiny_tokenizer):
    with pytest.raises(ValueError):
        extract_hidden_states(
            micro_model,
            tiny_tokenizer,
            ["a b"],
            pooling="max"
        )
