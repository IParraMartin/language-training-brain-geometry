"""Tests for the parity-export OOF assembly (no brainscore dependency).

The recording metric and full bundle require brainscore_language + real data and
are validated on the data box. Here we lock in the fail-closed out-of-fold
assembly logic, which is the piece most likely to silently misalign bundles.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain_code.experiments.pereira_parity_export import RecordedFolds, _assemble_oof


def _records(fold_test_ids, fold_preds):
    r = RecordedFolds()
    for ids, preds in zip(fold_test_ids, fold_preds):
        r.test_stimulus_ids.append(np.array(sorted(ids)))
        array = np.asarray(preds, dtype=np.float64)
        r.predictions.append(array)
        r.prediction_neuroid_ids.append(np.arange(array.shape[1]))
    return r


def test_oof_assembles_in_canonical_order():
    canonical = np.array(["s0", "s1", "s2", "s3"])
    # fold 0 tests s0,s1 ; fold 1 tests s2,s3 — predictions are per-sorted-test-id rows
    records = _records(
        [["s0", "s1"], ["s2", "s3"]],
        [[[10.0], [11.0]], [[12.0], [13.0]]]
    )
    oof, fold_ids = _assemble_oof(canonical, records)
    np.testing.assert_array_equal(oof[:, 0], [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(fold_ids, [0, 0, 1, 1])


def test_oof_raises_on_missing_coverage():
    canonical = np.array(["s0", "s1", "s2", "s3"])
    records = _records([["s0", "s1"]], [[[1.0], [2.0]]])  # s2,s3 never tested
    with pytest.raises(ValueError, match="unscored"):
        _assemble_oof(canonical, records)


def test_oof_raises_on_duplicate_fold_assignment():
    canonical = np.array(["s0", "s1"])
    # s0 appears as test in both folds → leakage / double assignment
    records = _records([["s0", "s1"], ["s0"]], [[[1.0], [2.0]], [[3.0]]])
    with pytest.raises(ValueError, match="more than one test fold"):
        _assemble_oof(canonical, records)


def test_oof_raises_on_unknown_test_stimulus():
    canonical = np.array(["s0", "s1"])
    records = _records([["s0", "sX"]], [[[1.0], [9.0]]])  # sX not in canonical ids
    with pytest.raises(ValueError, match="absent from targets"):
        _assemble_oof(canonical, records)
