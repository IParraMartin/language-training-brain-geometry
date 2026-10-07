from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from brain_code.experiments.data_loading import adapt_neural_assembly, build_contexts


def test_cumulative_context_resets_at_passage_boundary():
    contexts = build_contexts(
        ["one", "two", "three", "four"],
        np.array(["p1", "p1", "p2", "p2"]),
        context_mode="cumulative_passage"
    )
    assert contexts == ["one", "one two", "three", "three four"]


def test_cumulative_context_rejects_noncontiguous_groups():
    with pytest.raises(ValueError, match="non-contiguous"):
        build_contexts(
            ["a", "b", "c"],
            np.array(["p1", "p2", "p1"]),
            context_mode="cumulative_passage"
        )


def _assembly(transpose=False):
    values = np.arange(12, dtype=float).reshape(3, 4)
    assembly = xr.DataArray(
        values,
        dims=("presentation", "neuroid"),
        coords={
            "stimulus_id": ("presentation", ["s1", "s2", "s3"]),
            "sentence": ("presentation", ["A", "B", "C"]),
            # Pereira uses separate coordinates for context and CV grouping.
            "story": ("presentation", ["category-a", "category-a", "category-b"]),
            "passage_label": ("presentation", ["p1", "p1", "p2"]),
            "neuroid_id": ("neuroid", [1, 2, 3, 4])
        }
    )
    return assembly.transpose("neuroid", "presentation") if transpose else assembly


def test_neural_adapter_is_dimension_order_invariant():
    first = adapt_neural_assembly(_assembly(), "Pereira2018.243sentences-ridge")
    second = adapt_neural_assembly(_assembly(True), "Pereira2018.243sentences-ridge")
    np.testing.assert_array_equal(first.responses, second.responses)
    np.testing.assert_array_equal(first.stimulus_ids, second.stimulus_ids)


def test_unknown_neural_dimension_is_rejected():
    assembly = _assembly().expand_dims(subject=["subj1", "subj2"])
    with pytest.raises(RuntimeError, match="Unsupported neural dimension"):
        adapt_neural_assembly(assembly, "Pereira2018.243sentences-ridge")


def test_duplicate_stimulus_ids_are_rejected():
    assembly = _assembly().assign_coords(stimulus_id=("presentation", ["s1", "s1", "s3"]))
    with pytest.raises(RuntimeError, match="not unique"):
        adapt_neural_assembly(assembly, "Pereira2018.243sentences-ridge")


def test_grouping_must_match_official_split_coord():
    """A grouping coord other than the official split_coord must fail closed.

    Pereira2018 groups CV folds by ``story``. If only a finer/different coord is
    present (here ``passage_label``), the adapter must refuse rather than silently
    group differently from the official benchmark.
    """
    values = np.arange(12, dtype=float).reshape(3, 4)
    assembly = xr.DataArray(
        values,
        dims=("presentation", "neuroid"),
        coords={
            "stimulus_id": ("presentation", ["s1", "s2", "s3"]),
            "sentence": ("presentation", ["A", "B", "C"]),
            "passage_label": ("presentation", ["p1", "p1", "p2"]),  # present, but not `story`
            "neuroid_id": ("neuroid", [1, 2, 3, 4])
        }
    )
    with pytest.raises(RuntimeError, match="official benchmark split_coord 'story'"):
        adapt_neural_assembly(assembly, "Pereira2018.243sentences-ridge")
