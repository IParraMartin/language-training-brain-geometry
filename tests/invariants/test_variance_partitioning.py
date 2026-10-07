from __future__ import annotations

import numpy as np
import pytest

from brain_code.experiments.split_manifests import build_grouped_folds
from brain_code.experiments.variance_partitioning import (
    eval_encoding_model,
    eval_encoding_model_oof,
    validate_outer_folds
)


def _signal(seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(20), 4)
    x = rng.normal(size=(len(groups), 8))
    weights = rng.normal(size=(8, 3))
    y = 125.0 + x @ weights + rng.normal(scale=0.05, size=(len(groups), 3))
    import pandas as pd
    df = pd.DataFrame({"group": groups})
    folds = build_grouped_folds(
        df,
        "group",
        n_folds=5,
        seed=3
    )
    return (
        x,
        y,
        groups,
        folds
    )


def test_ridge_recovers_signal_with_large_intercept():
    x, y, groups, folds = _signal()
    result = eval_encoding_model_oof(
        x,
        y,
        folds,
        groups=groups,
        alpha_grid=np.array([1e-5, 1e-3, 0.1]),
        n_inner_folds=4
    )
    assert result["oof_r2_mean"] > 0.95
    assert result["oof_r_mean"] > 0.98
    assert result["predictions"].shape == y.shape


def test_ridge_is_stable_to_feature_rescaling():
    x, y, groups, folds = _signal(2)
    first = eval_encoding_model_oof(
        x,
        y,
        folds,
        groups=groups,
        alpha_grid=np.array([1e-4, 1e-2, 1.0]),
        n_inner_folds=4
    )
    scales = np.array([1e-4, 1e4, 3.0, 0.2, 17.0, 1.0, 0.01, 200.0])
    second = eval_encoding_model_oof(
        x * scales,
        y,
        folds,
        groups=groups,
        alpha_grid=np.array([1e-4, 1e-2, 1.0]),
        n_inner_folds=4
    )
    np.testing.assert_allclose(first["predictions"], second["predictions"], atol=1e-8)


def test_groups_are_required():
    x, y, _, _ = _signal()
    with pytest.raises(ValueError, match="groups_train is required"):
        eval_encoding_model(
            x[:40],
            y[:40],
            x[40:],
            y[40:]
        )


def test_outer_folds_require_exactly_one_test_prediction():
    _, _, groups, folds = _signal()
    broken = list(folds)
    duplicate_test = np.asarray(broken[-2][1])
    duplicate_train = np.setdiff1d(np.arange(len(groups)), duplicate_test)
    broken[-1] = (duplicate_train, duplicate_test)
    with pytest.raises(ValueError, match="exactly one test fold"):
        validate_outer_folds(broken, len(groups), groups)
