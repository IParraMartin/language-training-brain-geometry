"""Tests the neuroid coordinates on the custom-side strict-parity target assembly.

The pinned official metric (`brainscore_language...linear_predictivity.metric`)
has a fragile branch in `XarrayRegression._package_prediction`: when the target
carries exactly ONE neuroid coordinate it renames the ``neuroid`` dim to that
coordinate and re-stacks, raising "cannot create a new dimension with the same
name as an existing dimension". Official Pereira assemblies carry ~10 neuroid
coords and never hit it.

`scripts/export_pereira_parity._custom_target_assembly` adds a second neuroid
coordinate (``neuroid_number``) besides ``neuroid_id``, so the metric (which
collects target neuroid coords with the same `walk_coords` scan used below) sees
>= 2 and skips that branch, packaging predictions on the ``neuroid`` dim exactly
like the official path. This test pins that invariant with no network or model
download.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _fake_benchmark_data(n_stim=12, n_neuroid=5, seed=0):
    rng = np.random.RandomState(seed)
    return SimpleNamespace(
        raw_stimulus_ids=np.array([f"s{i}" for i in range(n_stim)]),
        sentences=np.array([f"sentence number {i}" for i in range(n_stim)], dtype="U"),
        groups=np.array([f"story{i % 3}" for i in range(n_stim)]),
        context_groups=np.array([f"passage{i % 4}" for i in range(n_stim)]),
        neuroid_ids=np.array([f"nid{j}" for j in range(n_neuroid)]),
        responses=rng.randn(n_stim, n_neuroid)
    )


def _neuroid_coord_names(assembly):
    """Lists neuroid coordinates exactly as the official metric does.

    The metric builds `_target_neuroid_values` by walking the coords whose dims
    include 'neuroid'.
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords

    return [name for name, dims, _ in walk_coords(assembly) if "neuroid" in dims]


def test_custom_target_assembly_exposes_multiple_neuroid_coords():
    """>= 2 neuroid coords => the metric's single-coord stacking branch is skipped."""
    pytest.importorskip("brainscore_language")  # NeuroidAssembly + walk_coords
    import scripts.export_pereira_parity as ex

    target = ex._custom_target_assembly(_fake_benchmark_data())
    neuroid_coords = _neuroid_coord_names(target)

    assert len(neuroid_coords) >= 2, (
        f"custom target must expose >= 2 neuroid coords so the official metric does "
        f"not hit the single-coord stacking bug; got {neuroid_coords}"
    )
    assert "neuroid_id" in neuroid_coords, neuroid_coords
    # the number of neuroids must be preserved regardless of the extra coord
    assert int(target.sizes["neuroid"]) == 5
