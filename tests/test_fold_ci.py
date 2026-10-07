"""Unit tests for the shared across-fold 95% CI helper."""

from __future__ import annotations

import numpy as np

from brain_code.experiments.fold_ci import fold_ci95, t975


def test_fold_ci95_center_matches_mean_and_is_symmetric():
    values = [0.10, 0.12, 0.08, 0.11, 0.09]
    ci = fold_ci95(values)
    assert ci["n_folds"] == 5
    assert abs(ci["mean"] - float(np.mean(values))) < 1e-12
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    # a t-based interval is symmetric about the mean
    assert abs((ci["ci_high"] - ci["mean"]) - (ci["mean"] - ci["ci_low"])) < 1e-9
    # half-width = t975(df) * sem
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    assert abs((ci["ci_high"] - ci["mean"]) - t975(4) * sem) < 1e-9


def test_fold_ci95_drops_nonfinite_folds():
    ci = fold_ci95([
        0.1,
        np.nan,
        0.2,
        np.inf
    ])
    assert ci["n_folds"] == 2
    assert abs(ci["mean"] - 0.15) < 1e-12


def test_fold_ci95_single_fold_has_no_interval():
    ci = fold_ci95([0.1])
    assert ci["n_folds"] == 1
    assert ci["mean"] == 0.1
    assert np.isnan(ci["ci_low"]) and np.isnan(ci["ci_high"])


def test_fold_ci95_empty_is_all_nan():
    ci = fold_ci95([])
    assert ci["n_folds"] == 0
    assert np.isnan(ci["mean"])


def test_t975_known_values_and_fallback():
    assert t975(4) == 2.776   # 5 folds
    assert t975(9) == 2.262   # official 10-fold path
    assert t975(200) == 1.96  # normal-approx fallback above the table
    assert np.isnan(t975(0))
