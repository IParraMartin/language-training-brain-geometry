"""Tests for manifold-geometry pressure metrics and PC ablation."""

from __future__ import annotations

import numpy as np
import pytest

from brain_code.experiments.manifold_pressure import (
    fit_pc_remover,
    participation_ratio,
    pc_ablation_curve,
    spectral_decay,
    geometry_alignment_correlation,
    grouped_ridge_fold_scorer,
    layerwise_spectral_profile,
    ridge_fold_scorer,
    spectral_profile
)


def _isotropic(n=2000, d=16, seed=0):
    """Near-isotropic Gaussian: every covariance eigenvalue equal in expectation."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def _rank_one(n=200, d=16, seed=0):
    """All rows on a single line through the origin."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(d)
    return np.outer(rng.standard_normal(n), direction)


def _powerlaw_reps(
    n=4000,
    d=64,
    alpha=1.0,
    seed=0
):
    """Gaussian whose covariance spectrum follows lambda_i ∝ i**(-alpha)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(1, d + 1, dtype=np.float64)
    scale = idx ** (-alpha / 2.0)          # std dev, so variance ∝ i**(-alpha)
    return rng.standard_normal((n, d)) * scale


def test_participation_ratio_of_isotropic_data_approaches_dimensionality():
    pr = participation_ratio(_isotropic(n=4000, d=16))
    assert pr == pytest.approx(16.0, rel=0.10)


def test_participation_ratio_of_rank_one_data_is_one():
    assert participation_ratio(_rank_one()) == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_is_scale_invariant():
    X = _powerlaw_reps(d=32)
    assert participation_ratio(X * 137.0) == pytest.approx(participation_ratio(X), rel=1e-9)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.8])
def test_spectral_decay_recovers_known_powerlaw_exponent(alpha):
    """A covariance spectrum built as i**(-alpha) must fit back to alpha."""
    X = _powerlaw_reps(
        n=8000,
        d=64,
        alpha=alpha,
        seed=1
    )
    assert spectral_decay(X) == pytest.approx(alpha, abs=0.15)


def test_spectral_decay_of_isotropic_data_is_near_zero():
    """A flat spectrum has no decay."""
    assert spectral_decay(_isotropic(n=8000, d=32)) == pytest.approx(0.0, abs=0.2)


def test_spectral_profile_reports_all_four_geometry_metrics():
    profile = spectral_profile(_powerlaw_reps())
    assert set(profile) == {
        "effective_rank",
        "anisotropy",
        "participation_ratio",
        "spectral_decay",
        "n_samples",
        "n_features"
    }


def test_spectral_profile_orders_dimensionality_by_construction():
    """Steeper spectral decay must mean lower effective dimensionality."""
    flat = spectral_profile(_powerlaw_reps(alpha=0.3, seed=2))
    steep = spectral_profile(_powerlaw_reps(alpha=2.5, seed=2))
    assert steep["participation_ratio"] < flat["participation_ratio"]
    assert steep["effective_rank"] < flat["effective_rank"]
    assert steep["spectral_decay"] > flat["spectral_decay"]


def test_pc_remover_removes_exactly_the_requested_number_of_directions():
    X = _powerlaw_reps(n=500, d=32, seed=3)
    out = fit_pc_remover(X, 5).transform(X)
    # Rank drops by exactly 5 (mean-centering costs nothing extra here).
    assert np.linalg.matrix_rank(out, tol=1e-8) == 32 - 5


def test_pc_remover_with_zero_components_is_a_no_op_up_to_centering():
    X = _powerlaw_reps(n=500, d=16, seed=4)
    out = fit_pc_remover(X, 0).transform(X)
    np.testing.assert_allclose(out, X - X.mean(axis=0), atol=1e-9)


def test_pc_remover_fit_on_train_is_applied_unchanged_to_held_out_rows():
    """The projector must be a fixed linear map, not refit per input matrix."""
    X = _powerlaw_reps(n=600, d=24, seed=5)
    remover = fit_pc_remover(X[:400], 3)
    held_out = X[400:]
    # Transforming the held-out block alone must equal transforming it in context.
    both = remover.transform(X)[400:]
    np.testing.assert_allclose(remover.transform(held_out), both, atol=1e-9)


def test_pc_remover_leaves_variance_in_the_directions_it_did_not_remove():
    X = _powerlaw_reps(
        n=800,
        d=32,
        alpha=1.5,
        seed=6
    )
    remover = fit_pc_remover(X, 4)
    out = remover.transform(X)
    assert out.var(axis=0).sum() > 0.0
    assert out.var(axis=0).sum() < X.var(axis=0).sum()


def _identity_fold_scorer(
    X_tr,
    Y_tr,
    X_te,
    Y_te,
    **_
):
    """Least-squares readout; returns test-set predictions."""
    beta, *_ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
    return X_te @ beta


def _two_folds(n):
    half = n // 2
    a, b = np.arange(half), np.arange(half, n)
    return [(a, b), (b, a)]


def test_pc_ablation_curve_returns_one_entry_per_requested_m():
    rng = np.random.default_rng(7)
    X = _powerlaw_reps(n=200, d=16, seed=7)
    Y = X @ rng.standard_normal((16, 4)) + 0.1 * rng.standard_normal((200, 4))
    curve = pc_ablation_curve(
        X,
        Y,
        _two_folds(200),
        m_grid=[0, 1, 4],
        fold_scorer=_identity_fold_scorer
    )
    assert [row["m"] for row in curve] == [0, 1, 4]


def test_pc_ablation_curve_degrades_as_more_signal_directions_are_removed():
    """Y is driven by X's leading PCs, so ablating them must cost alignment."""
    rng = np.random.default_rng(8)
    X = _powerlaw_reps(
        n=400,
        d=16,
        alpha=2.0,
        seed=8
    )
    Y = X @ rng.standard_normal((16, 3)) + 0.05 * rng.standard_normal((400, 3))
    curve = pc_ablation_curve(
        X,
        Y,
        _two_folds(400),
        m_grid=[0, 2, 8],
        fold_scorer=_identity_fold_scorer
    )
    scores = [row["score"] for row in curve]
    assert scores[0] > scores[1] > scores[2]


def test_pc_ablation_curve_reports_variance_removed_at_each_step():
    X = _powerlaw_reps(
        n=300,
        d=16,
        alpha=1.5,
        seed=9
    )
    Y = X[:, :2].copy()
    curve = pc_ablation_curve(
        X,
        Y,
        _two_folds(300),
        m_grid=[0, 3],
        fold_scorer=_identity_fold_scorer
    )
    assert curve[0]["variance_removed"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < curve[1]["variance_removed"] < 1.0


def test_pc_ablation_curve_refits_the_projector_inside_each_fold():
    """A projector fit on all rows would leak held-out structure into the score."""
    seen = []

    def spy_scorer(
        X_tr,
        Y_tr,
        X_te,
        Y_te,
        **_
    ):
        seen.append((X_tr.shape[0], X_te.shape[0]))
        return _identity_fold_scorer(
            X_tr,
            Y_tr,
            X_te,
            Y_te
        )

    X = _powerlaw_reps(n=200, d=16, seed=10)
    Y = X[:, :2].copy()
    pc_ablation_curve(
        X,
        Y,
        _two_folds(200),
        m_grid=[2],
        fold_scorer=spy_scorer
    )
    # One scorer call per fold, each seeing only its own split.
    assert seen == [(100, 100), (100, 100)]


def test_ridge_fold_scorer_returns_one_prediction_row_per_test_stimulus():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((120, 8))
    Y = X @ rng.standard_normal((8, 5))
    scorer = ridge_fold_scorer(alpha_grid=[1.0, 10.0], n_inner_folds=2)
    pred = scorer(
        X[:80],
        Y[:80],
        X[80:],
        Y[80:]
    )
    assert pred.shape == (40, 5)


def test_ridge_fold_scorer_recovers_a_clean_linear_map():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((200, 6))
    Y = X @ rng.standard_normal((6, 3)) + 0.01 * rng.standard_normal((200, 3))
    scorer = ridge_fold_scorer(alpha_grid=[1e-3, 1e-1], n_inner_folds=2)
    pred = scorer(
        X[:150],
        Y[:150],
        X[150:],
        Y[150:]
    )
    assert np.corrcoef(pred.ravel(), Y[150:].ravel())[0, 1] > 0.95


def test_ridge_fold_scorer_fits_an_intercept():
    """A constant offset in Y must not be charged to the coefficients."""
    rng = np.random.default_rng(13)
    X = rng.standard_normal((150, 4))
    Y = X @ rng.standard_normal((4, 2)) + 50.0
    scorer = ridge_fold_scorer(alpha_grid=[1.0], n_inner_folds=2)
    pred = scorer(
        X[:100],
        Y[:100],
        X[100:],
        Y[100:]
    )
    assert np.abs(pred.mean() - 50.0) < 5.0


def test_ridge_fold_scorer_ignores_held_out_targets():
    """Y_te is passed for signature symmetry only; it must never be fit on."""
    rng = np.random.default_rng(14)
    X = rng.standard_normal((100, 5))
    Y = X @ rng.standard_normal((5, 2))
    scorer = ridge_fold_scorer(alpha_grid=[1.0], n_inner_folds=2)
    honest = scorer(
        X[:70],
        Y[:70],
        X[70:],
        Y[70:]
    )
    corrupted = scorer(
        X[:70],
        Y[:70],
        X[70:],
        rng.standard_normal((30, 2)) * 100
    )
    np.testing.assert_allclose(honest, corrupted, atol=1e-12)


def test_layerwise_spectral_profile_returns_one_row_per_layer():
    reps = {
        i: _powerlaw_reps(
            n=200,
            d=32,
            alpha=0.5 + 0.1 * i,
            seed=i
        )
        for i in range(4)
    }
    rows = layerwise_spectral_profile(reps)
    assert [row["layer"] for row in rows] == [0, 1, 2, 3]
    assert all("participation_ratio" in row for row in rows)


def test_layerwise_spectral_profile_sorts_layers_numerically():
    reps = {10: _isotropic(n=100, d=8), 2: _isotropic(n=100, d=8, seed=1)}
    assert [row["layer"] for row in layerwise_spectral_profile(reps)] == [2, 10]


def test_geometry_alignment_correlation_detects_a_positive_coupling():
    """Alignment rising with participation ratio must give rho ≈ +1."""
    profiles = [
        {"layer": i, "participation_ratio": float(i), "anisotropy": float(-i)}
        for i in range(6)
    ]
    scores = [0.10, 0.12, 0.15, 0.19, 0.22, 0.30]
    corr = geometry_alignment_correlation(profiles, scores)
    assert corr["participation_ratio"]["rho"] == pytest.approx(1.0, abs=1e-9)
    assert corr["anisotropy"]["rho"] == pytest.approx(-1.0, abs=1e-9)


def test_geometry_alignment_correlation_skips_non_metric_fields():
    profiles = [
        {
            "layer": i,
            "n_samples": 10,
            "n_features": 4,
            "anisotropy": float(i)
        }
        for i in range(5)
    ]
    corr = geometry_alignment_correlation(profiles, [1, 2, 3, 4, 5])
    assert set(corr) == {"anisotropy"}


def test_geometry_alignment_correlation_requires_matching_lengths():
    profiles = [{"layer": 0, "anisotropy": 1.0}]
    with pytest.raises(ValueError, match="length"):
        geometry_alignment_correlation(profiles, [0.1, 0.2])


def test_geometry_alignment_correlation_never_correlates_alignment_with_itself():
    """The runner stores alignment_r inside each profile row; it is not a geometry metric."""
    profiles = [
        {"layer": i, "anisotropy": float(i), "alignment_r": 0.1 * i} for i in range(5)
    ]
    corr = geometry_alignment_correlation(profiles, [0.0, 0.1, 0.2, 0.3, 0.4])
    assert "alignment_r" not in corr


def test_pc_ablation_curve_scores_per_fold_rather_than_pooling_predictions():
    """Per-story fMRI offsets differ across folds; pooling drives Pearson r to 0.

    variance_partitioning.eval_encoding_model_oof averages each neuroid's
    per-fold score instead of scoring concatenated predictions, and this curve
    must use the same convention or its m=0 baseline will not match the layer
    sweep it is compared against.
    """
    rng = np.random.default_rng(20)
    n, d = 200, 8
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, 3))
    Y = X @ W
    # Fold B's block sits at a large constant offset, as fMRI blocks do.
    Y[100:] += 500.0

    def offset_blind_scorer(
        X_tr,
        Y_tr,
        X_te,
        Y_te,
        **_
    ):
        beta, *_rest = np.linalg.lstsq(X_tr - X_tr.mean(0), Y_tr - Y_tr.mean(0), rcond=None)
        return (X_te - X_tr.mean(0)) @ beta + Y_tr.mean(0)

    curve = pc_ablation_curve(
        X,
        Y,
        _two_folds(n),
        m_grid=[0],
        fold_scorer=offset_blind_scorer
    )
    # Within each fold the readout is near-perfect, so the per-fold average is ~1.
    # Pooling the two blocks against a global mean would collapse this.
    assert curve[0]["score"] > 0.95


def test_pc_ablation_baseline_matches_the_layer_sweep_scorer():
    """m=0 must reproduce subspace_erasure._score_condition on the same inputs.

    The ablation curve is reported as a fraction of the layer's alignment, so a
    baseline computed by a different scorer would make every retained-% wrong.
    """
    from brain_code.experiments.subspace_erasure import _score_condition

    rng = np.random.default_rng(21)
    n_groups, per_group, d = 8, 12, 10
    n = n_groups * per_group
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.standard_normal((n, d))
    Y = X @ rng.standard_normal((d, 6)) + 0.3 * rng.standard_normal((n, 6))

    folds = [
        (np.where(groups % 4 != k)[0], np.where(groups % 4 == k)[0]) for k in range(4)
    ]
    alpha_grid = np.array([1.0, 10.0, 100.0])

    _, _, sweep_r = _score_condition(
        X,
        Y,
        folds,
        alpha_grid=alpha_grid,
        n_inner_folds=2,
        groups=groups
    )
    curve = pc_ablation_curve(
        X,
        Y,
        folds,
        [0],
        fold_scorer=grouped_ridge_fold_scorer(alpha_grid, n_inner_folds=2, groups=groups)
    )
    assert curve[0]["score"] == pytest.approx(sweep_r, abs=1e-9)


def test_grouped_ridge_fold_scorer_receives_the_indices_it_needs_for_grouping():
    """The scorer slices `groups` by fold index, so the curve must pass them."""
    rng = np.random.default_rng(22)
    n, d = 60, 5
    groups = np.repeat(np.arange(10), 6)
    X = rng.standard_normal((n, d))
    Y = X @ rng.standard_normal((d, 2))
    folds = [(np.arange(30), np.arange(30, 60)), (np.arange(30, 60), np.arange(30))]
    scorer = grouped_ridge_fold_scorer(np.array([1.0]), n_inner_folds=2, groups=groups)
    curve = pc_ablation_curve(
        X,
        Y,
        folds,
        [0, 2],
        fold_scorer=scorer
    )
    assert len(curve) == 2 and np.isfinite(curve[0]["score"])
