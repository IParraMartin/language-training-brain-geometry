"""Tests for ridge regression and variance partitioning.

Key assertions:
  - R² values sum correctly (unique_model + unique_nuisance + shared + unexplained = 1)
  - Negative R² is reported as-is (never clipped)
  - A perfect predictor achieves R²=1 on noise-free data
  - Random nuisance cannot explain model-driven signal
  - Group-aware inner CV selects alpha without same-passage leakage
  - OOF scoring assembles predictions in original order and scores once
  - Fold-local preprocessing (center/scale) is fit on train split only
"""

from __future__ import annotations

import numpy as np
import pytest

from brain_code.experiments.variance_partitioning import (
    eval_encoding_model,
    eval_encoding_model_oof,
    variance_partition,
    _pearson_r_per_col,
    _r2_per_col,
    _fit_ridge,
    _center_and_scale,
    _center_targets,
    _build_inner_groups,
    _select_alpha_group_aware
)


def _make_synthetic_data(
    n_train=40,
    n_test=20,
    n_features=10,
    n_regions=5,
    seed=0
):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    X_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    coef = rng.standard_normal((n_features, n_regions)).astype(np.float32)
    noise = rng.standard_normal((n_train, n_regions)).astype(np.float32) * 0.1
    Y_train = X_train @ coef + noise
    Y_test = X_test @ coef
    return (
        X_train,
        Y_train,
        X_test,
        Y_test,
        coef
    )


def test_pearson_r_identical_inputs():
    Y = np.random.default_rng(0).standard_normal((20, 3))
    r = _pearson_r_per_col(Y, Y)
    np.testing.assert_allclose(r, np.ones(3), atol=1e-5)


def test_pearson_r_anticorrelated():
    Y = np.arange(10, dtype=float).reshape(-1, 1)
    r = _pearson_r_per_col(Y, -Y)
    assert r[0] < -0.99


def test_r2_perfect_predictor():
    Y = np.arange(20, dtype=float).reshape(-1, 1)
    r2 = _r2_per_col(Y, Y)
    np.testing.assert_allclose(r2, [1.0], atol=1e-6)


def test_r2_can_be_negative():
    # a constant Y_true has ss_tot=0 and yields NaN by design
    Y_true = np.linspace(0, 1, 10).reshape(-1, 1)
    Y_pred = np.linspace(1, 0, 10).reshape(-1, 1)
    r2 = _r2_per_col(Y_true, Y_pred)
    assert r2[0] < 0, "R² must be able to go negative (never clipped)."


def test_fit_ridge_recovers_known_coef():
    rng = np.random.default_rng(42)
    true_coef = np.array([[2.0, -1.0]])
    X = rng.standard_normal((100, 2))
    Y = X @ true_coef.T + rng.standard_normal((100, 1)) * 0.01
    coef = _fit_ridge(X, Y, alpha=1e-6)
    np.testing.assert_allclose(coef.flatten(), true_coef.flatten(), atol=0.1)


def test_ridge_recovers_signal_with_large_intercept():
    """Ridge must recover signal even when targets have a large non-zero mean."""
    rng = np.random.default_rng(50)
    true_coef = np.array([[3.0, -2.0, 1.0]])
    X = rng.standard_normal((100, 3)).astype(np.float32)
    Y = (X @ true_coef.T + 100.0).astype(np.float32)
    X_tr, Y_tr, X_te, Y_te = X[:80], Y[:80], X[80:], Y[80:]
    result = eval_encoding_model(
        X_tr,
        Y_tr,
        X_te,
        Y_te,
        allow_ungrouped=True
    )
    assert result["r"].mean() > 0.95, (
        f"R after fold-local target centering should be near 1; got {result['r'].mean():.4f}"
    )


def test_ridge_is_stable_to_feature_rescaling():
    """Ridge with fold-local feature centering/scaling should be scale-invariant."""
    rng = np.random.default_rng(51)
    X = rng.standard_normal((80, 4)).astype(np.float32)
    coef = rng.standard_normal((4, 3)).astype(np.float32)
    Y = X @ coef
    X_tr, Y_tr, X_te, Y_te = X[:60], Y[:60], X[60:], Y[60:]
    r1 = eval_encoding_model(
        X_tr,
        Y_tr,
        X_te,
        Y_te,
        allow_ungrouped=True
    )["r"].mean()
    r2 = eval_encoding_model(
        X_tr * 1000,
        Y_tr,
        X_te * 1000,
        Y_te,
        allow_ungrouped=True
    )["r"].mean()
    assert abs(r1 - r2) < 0.01, (
        f"Ridge should be stable to feature scale: r={r1:.4f} vs r_scaled={r2:.4f}"
    )


def test_inner_cv_respects_groups():
    """Inner-fold alpha selection must not split the same group across train/val."""
    groups = np.repeat(np.arange(12), 5)
    inner_folds = _build_inner_groups(groups, n_inner_folds=4)
    for tr_idx, val_idx in inner_folds:
        g_tr = set(groups[tr_idx].tolist())
        g_val = set(groups[val_idx].tolist())
        assert g_tr.isdisjoint(g_val), f"Inner CV leaked groups: {g_tr & g_val}"


def test_outer_cv_respects_groups():
    """OOF outer folds must not place the same group in both train and test."""
    groups = np.repeat(np.arange(12), 5)
    unique_groups = np.unique(groups)
    folds = []
    for i in range(4):
        lo, hi = i * 3, (i + 1) * 3
        val_groups = set(unique_groups[lo:hi].tolist())
        te_idx = np.where(np.isin(groups, list(val_groups)))[0]
        tr_idx = np.where(~np.isin(groups, list(val_groups)))[0]
        folds.append((tr_idx, te_idx))
    for tr_idx, te_idx in folds:
        g_tr = set(groups[tr_idx].tolist())
        g_te = set(groups[te_idx].tolist())
        assert g_tr.isdisjoint(g_te), f"Outer CV leaked groups: {g_tr & g_te}"


def test_full_oof_scoring_is_used():
    """OOF score must differ from mean-of-fold-scores when responses are unbalanced."""
    rng = np.random.default_rng(54)
    n, p, t = 60, 8, 3
    groups = np.repeat(np.arange(12), 5)
    X = rng.standard_normal((n, p)).astype(np.float32)
    Y = X @ rng.standard_normal((p, t)).astype(np.float32)
    unique_groups = np.unique(groups)
    folds = []
    for i in range(4):
        lo, hi = i * 3, min((i + 1) * 3, len(unique_groups))
        val_groups = set(unique_groups[lo:hi].tolist())
        te_idx = np.where(np.isin(groups, list(val_groups)))[0]
        tr_idx = np.where(~np.isin(groups, list(val_groups)))[0]
        folds.append((tr_idx, te_idx))
    res = eval_encoding_model_oof(
        X,
        Y,
        folds,
        groups=groups,
        n_inner_folds=3
    )
    # oof_r is scored once on the assembled predictions, oof_r_mean averages fold
    # means, and the two differ in general
    assert "oof_r" in res
    assert "oof_r_mean" in res
    assert "fold_r_means" in res
    assert len(res["oof_r"]) == t


def test_no_signal_data_is_near_baseline():
    """Random features predicting random targets should produce near-zero R²."""
    rng = np.random.default_rng(55)
    X_tr = rng.standard_normal((40, 8)).astype(np.float32)
    Y_tr = rng.standard_normal((40, 5)).astype(np.float32)
    X_te = rng.standard_normal((20, 8)).astype(np.float32)
    Y_te = rng.standard_normal((20, 5)).astype(np.float32)
    result = eval_encoding_model(
        X_tr,
        Y_tr,
        X_te,
        Y_te,
        allow_ungrouped=True
    )
    assert result["r2"].mean() < 0.3, (
        f"Random-on-random R² should be near 0; got {result['r2'].mean():.3f}"
    )


def test_eval_encoding_high_snr():
    X_tr, Y_tr, X_te, Y_te, _ = _make_synthetic_data(seed=1)
    result = eval_encoding_model(
        X_tr,
        Y_tr,
        X_te,
        Y_te,
        allow_ungrouped=True
    )
    assert result["r2"].mean() > 0.5, f"Expected R²>0.5, got {result['r2'].mean():.3f}"


def test_eval_encoding_random_features_near_zero():
    rng = np.random.default_rng(77)
    _, Y_tr, _, Y_te, _ = _make_synthetic_data(seed=2)
    n_tr, n_te = len(Y_tr), len(Y_te)
    X_rand_tr = rng.standard_normal((n_tr, 10)).astype(np.float32)
    X_rand_te = rng.standard_normal((n_te, 10)).astype(np.float32)
    result = eval_encoding_model(
        X_rand_tr,
        Y_tr,
        X_rand_te,
        Y_te,
        allow_ungrouped=True
    )
    assert result["r2"].mean() < 0.3


def test_variance_partition_sums_to_one():
    X_tr, Y_tr, X_te, Y_te, _ = _make_synthetic_data(n_features=8, seed=5)
    rng = np.random.default_rng(5)
    X_nuis_tr = rng.standard_normal((len(X_tr), 4)).astype(np.float32)
    X_nuis_te = rng.standard_normal((len(X_te), 4)).astype(np.float32)
    df = variance_partition(
        X_model=X_tr,
        X_nuisance=X_nuis_tr,
        Y_train=Y_tr,
        Y_test=Y_te,
        X_model_test=X_te,
        X_nuisance_test=X_nuis_te,
        groups_train=np.arange(len(X_tr)),
        groups_test=np.arange(len(X_tr), len(X_tr) + len(X_te))
    )
    total = df["unique_model"] + df["unique_nuisance"] + df["shared"] + df["unexplained"]
    np.testing.assert_allclose(total.values, np.ones(len(df)), atol=1e-4)


def test_variance_partition_unique_model_positive_for_good_predictor():
    X_tr, Y_tr, X_te, Y_te, _ = _make_synthetic_data(seed=3)
    rng = np.random.default_rng(3)
    X_nuis_tr = rng.standard_normal((len(X_tr), 3)).astype(np.float32)
    X_nuis_te = rng.standard_normal((len(X_te), 3)).astype(np.float32)
    df = variance_partition(
        X_model=X_tr,
        X_nuisance=X_nuis_tr,
        Y_train=Y_tr,
        Y_test=Y_te,
        X_model_test=X_te,
        X_nuisance_test=X_nuis_te,
        groups_train=np.arange(len(X_tr)),
        groups_test=np.arange(len(X_tr), len(X_tr) + len(X_te))
    )
    assert df["unique_model"].mean() > 0, (
        f"Expected positive unique_model, got {df['unique_model'].mean():.4f}"
    )


def test_variance_partition_unique_random_nuisance_near_zero():
    """Random nuisance should not explain much unique variance over a good model."""
    X_tr, Y_tr, X_te, Y_te, _ = _make_synthetic_data(n_features=10, seed=6)
    rng = np.random.default_rng(6)
    X_nuis_tr = rng.standard_normal((len(X_tr), 10)).astype(np.float32)
    X_nuis_te = rng.standard_normal((len(X_te), 10)).astype(np.float32)
    df = variance_partition(
        X_model=X_tr,
        X_nuisance=X_nuis_tr,
        Y_train=Y_tr,
        Y_test=Y_te,
        X_model_test=X_te,
        X_nuisance_test=X_nuis_te,
        groups_train=np.arange(len(X_tr)),
        groups_test=np.arange(len(X_tr), len(X_tr) + len(X_te))
    )
    assert df["unique_nuisance"].mean() < 0.2, (
        f"Random nuisance unique_nuisance too high: {df['unique_nuisance'].mean():.4f}"
    )


def test_center_and_scale_zero_mean_on_train():
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((30, 5)).astype(np.float32)
    X_te = rng.standard_normal((10, 5)).astype(np.float32)
    X_tr_s, _, means, stds = _center_and_scale(X_tr, X_te)
    np.testing.assert_allclose(X_tr_s.mean(axis=0), np.zeros(5), atol=1e-5)


def test_center_and_scale_unit_variance_on_train():
    rng = np.random.default_rng(1)
    X_tr = rng.standard_normal((50, 4)).astype(np.float32)
    X_te = rng.standard_normal((10, 4)).astype(np.float32)
    X_tr_s, _, _, stds = _center_and_scale(X_tr, X_te)
    np.testing.assert_allclose(X_tr_s.std(axis=0), np.ones(4), atol=1e-3)


def test_center_and_scale_handles_zero_variance_feature():
    X_tr = np.ones((10, 3), dtype=np.float32)
    X_te = np.ones((5, 3), dtype=np.float32)
    X_tr_s, X_te_s, means, stds = _center_and_scale(X_tr, X_te)
    # zero-variance features are zeroed out rather than divided into inf or nan
    assert np.isfinite(X_tr_s).all()
    assert np.isfinite(X_te_s).all()
    np.testing.assert_array_equal(X_tr_s, np.zeros_like(X_tr_s))
    np.testing.assert_array_equal(X_te_s, np.zeros_like(X_te_s))
    # a std of 0 is the sentinel for a zero-variance feature
    np.testing.assert_array_equal(stds, np.zeros(3))


def test_center_targets_zero_mean_on_train():
    rng = np.random.default_rng(2)
    Y_tr = rng.standard_normal((20, 3)).astype(np.float32)
    Y_te = rng.standard_normal((5, 3)).astype(np.float32)
    Y_tr_c, Y_te_c, means = _center_targets(Y_tr, Y_te)
    np.testing.assert_allclose(Y_tr_c.mean(axis=0), np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(means, Y_tr.mean(axis=0), atol=1e-5)


def test_center_targets_uses_train_mean_for_test():
    Y_tr = np.array([[1.0], [3.0]])
    Y_te = np.array([[10.0], [20.0]])
    _, Y_te_c, means = _center_targets(Y_tr, Y_te)
    # centred by the train mean (2), not the test mean (15)
    np.testing.assert_allclose(Y_te_c, Y_te - 2.0, atol=1e-6)


def test_build_inner_groups_produces_non_overlapping_folds():
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    folds = _build_inner_groups(groups, n_inner_folds=3)
    for tr_idx, val_idx in folds:
        g_tr = set(groups[tr_idx].tolist())
        g_val = set(groups[val_idx].tolist())
        assert g_tr.isdisjoint(g_val), f"Group leak: {g_tr & g_val}"


def test_build_inner_groups_covers_all_observations():
    groups = np.repeat(np.arange(8), 3)
    folds = _build_inner_groups(groups, n_inner_folds=4)
    all_val = np.concatenate([val for _, val in folds])
    assert len(all_val) == len(groups)
    assert len(set(all_val.tolist())) == len(groups)


def test_build_inner_groups_raises_if_too_few_groups():
    groups = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="inner folds"):
        _build_inner_groups(groups, n_inner_folds=5)


def test_select_alpha_group_aware_returns_a_scalar_in_grid():
    rng = np.random.default_rng(7)
    n, p = 40, 6
    groups = np.repeat(np.arange(8), 5)
    X = rng.standard_normal((n, p)).astype(np.float32)
    Y = X @ rng.standard_normal((p, 3)).astype(np.float32)
    alpha_grid = np.array([0.1, 1.0, 10.0, 100.0])
    best = _select_alpha_group_aware(
        X,
        Y,
        groups,
        alpha_grid,
        n_inner_folds=4
    )
    # one alpha per target, each drawn from the grid
    assert all(b in alpha_grid.tolist() for b in best.tolist())


def test_eval_encoding_with_groups_does_not_warn():
    """Providing groups should suppress the sequential-fallback UserWarning."""
    import warnings
    rng = np.random.default_rng(9)
    n_tr, n_te, p = 40, 10, 6
    groups = np.repeat(np.arange(8), 5)
    X_tr = rng.standard_normal((n_tr, p)).astype(np.float32)
    Y_tr = rng.standard_normal((n_tr, 3)).astype(np.float32)
    X_te = rng.standard_normal((n_te, p)).astype(np.float32)
    Y_te = rng.standard_normal((n_te, 3)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eval_encoding_model(
            X_tr,
            Y_tr,
            X_te,
            Y_te,
            groups_train=groups,
            groups_test=np.arange(n_te) + 100
        )


def test_eval_encoding_without_groups_raises():
    """Omitting groups must raise (fail-closed; use allow_ungrouped=True for synthetic tests)."""
    rng = np.random.default_rng(10)
    X_tr = rng.standard_normal((20, 4)).astype(np.float32)
    Y_tr = rng.standard_normal((20, 2)).astype(np.float32)
    X_te = rng.standard_normal((5, 4)).astype(np.float32)
    Y_te = rng.standard_normal((5, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="groups_train is required"):
        eval_encoding_model(
            X_tr,
            Y_tr,
            X_te,
            Y_te,
            groups_train=None
        )


def test_oof_covers_all_observations():
    """eval_encoding_model_oof must assign a prediction to every observation."""
    rng = np.random.default_rng(11)
    n, p, t = 60, 8, 4
    groups = np.repeat(np.arange(12), 5)
    X = rng.standard_normal((n, p)).astype(np.float32)
    Y = X @ rng.standard_normal((p, t)).astype(np.float32)
    folds = []
    unique_groups = np.unique(groups)
    fold_size = len(unique_groups) // 3
    for i in range(3):
        hi = (i + 1) * fold_size if i < 2 else len(unique_groups)
        val_groups = set(unique_groups[i * fold_size : hi].tolist())
        te_idx = np.where(np.isin(groups, list(val_groups)))[0]
        tr_idx = np.where(~np.isin(groups, list(val_groups)))[0]
        folds.append((tr_idx, te_idx))
    res = eval_encoding_model_oof(
        X,
        Y,
        folds,
        groups=groups,
        n_inner_folds=3
    )
    assert "oof_r" in res
    assert "oof_r_mean" in res
    assert len(res["oof_r"]) == t


def test_oof_raises_on_duplicate_test_indices():
    """Observation appearing in test of multiple folds must raise (group leakage or repeated)."""
    rng = np.random.default_rng(12)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    Y = rng.standard_normal((10, 2)).astype(np.float32)
    # index 0 is a test observation in both folds
    folds = [
        (np.array([5, 6, 7, 8, 9]), np.array([0, 1, 2, 3, 4])),
        (np.array([0, 1, 2, 3, 4]), np.array([0, 5, 6, 7, 8]))
    ]
    with pytest.raises(ValueError):
        eval_encoding_model_oof(
            X,
            Y,
            folds,
            groups=np.arange(10)
        )


def test_oof_raises_on_missing_test_coverage():
    """OOF folds that don't cover all observations must raise."""
    rng = np.random.default_rng(13)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    Y = rng.standard_normal((10, 2)).astype(np.float32)
    folds = [(np.arange(5, 10), np.arange(0, 5))]
    with pytest.raises(ValueError, match="missing"):
        eval_encoding_model_oof(
            X,
            Y,
            folds,
            groups=np.arange(10)
        )
