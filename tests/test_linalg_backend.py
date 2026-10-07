"""Tests for the GPU-accelerated linear algebra backend.

The point of these tests is that acceleration must not change results. Every
optimization here (eigendecomposed alpha paths, Gram-matrix spectra, CUDA
dispatch) is an algebraic identity, so the accelerated path must agree with the
straightforward reference implementation.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from brain_code.experiments import linalg_backend as B
from brain_code.experiments.geometry import anisotropy, effective_rank, spectrum
from brain_code.experiments.variance_partitioning import _DEFAULT_ALPHA_GRID, _pearson_r_per_col, _r2_per_col


def _reference_cv_score(
    X,
    Y,
    alpha,
    n_folds,
    criterion
):
    """Cross-validated score for one alpha, via the direct-solve reference."""
    n = len(X)
    fold_size = n // n_folds
    total = 0.0
    for fold_i in range(n_folds):
        lo = fold_i * fold_size
        hi = lo + fold_size if fold_i < n_folds - 1 else n
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False
        X_tr, Y_tr, X_val, Y_val = X[mask], Y[mask], X[~mask], Y[~mask]
        nn, pp = X_tr.shape
        if nn >= pp:
            A = X_tr.T @ X_tr
            A[np.diag_indices(pp)] += alpha
            coef = np.linalg.solve(A, X_tr.T @ Y_tr)
        else:
            A = X_tr @ X_tr.T
            A[np.diag_indices(nn)] += alpha
            coef = X_tr.T @ np.linalg.solve(A, Y_tr)
        pred = X_val @ coef
        score = (
            _r2_per_col(Y_val, pred) if criterion == "r2"
            else _pearson_r_per_col(Y_val, pred)
        )
        total += float(score.mean())
    return total


def _reference_select_alpha(
    X,
    Y,
    alpha_grid,
    n_folds,
    criterion="r2"
):
    """Selects alpha by direct-solve cross-validation, the reference the backend must match."""
    scores = [
        _reference_cv_score(
            X,
            Y,
            alpha,
            n_folds,
            criterion
        )
        for alpha in alpha_grid
    ]
    return float(alpha_grid[np.argmax(scores)])


@pytest.mark.parametrize("n,p,r", [(60, 20, 8), (40, 120, 5), (80, 80, 12)])
@pytest.mark.parametrize("criterion", ["r2", "r"])
def test_select_alpha_matches_reference(
    n,
    p,
    r,
    criterion
):
    """The backend picks the same alpha as a direct-solve loop, per criterion."""
    rng = np.random.default_rng(n + p + r)
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, r))
    assert B.select_alpha(
        X,
        Y,
        _DEFAULT_ALPHA_GRID,
        4,
        criterion=criterion
    ) == _reference_select_alpha(
        X,
        Y,
        _DEFAULT_ALPHA_GRID,
        4,
        criterion=criterion
    )


def test_default_criterion_is_r2():
    """The default matches the reported metric (R²), not Pearson r."""
    rng = np.random.default_rng(11)
    X, Y = rng.standard_normal((70, 25)), rng.standard_normal((70, 6))
    assert B.select_alpha(
        X,
        Y,
        _DEFAULT_ALPHA_GRID,
        4
    ) == B.select_alpha(
        X,
        Y,
        _DEFAULT_ALPHA_GRID,
        4,
        criterion="r2"
    )


def test_r2_criterion_maximizes_cv_r2():
    """The 'r2' criterion maximizes the CV R² objective it optimizes.

    This is the property that matters and the only one that holds in general:
    selecting on R² gives cross-validated R² at least as high as any other alpha
    in the grid, including whichever alpha the 'r' criterion picks. Note it is
    *not* true that R² selection always chooses a stronger penalty — on random
    problems it frequently chooses a weaker one. The large improvement seen on
    the real Pereira data (test R² -0.31 -> -0.04) is an empirical result, not a
    general property of the criterion.
    """
    rng = np.random.default_rng(4)
    for _ in range(6):
        n = int(rng.integers(60, 160))
        X = rng.standard_normal((n, 40))
        Y = X @ rng.standard_normal((40, 10)) * 0.1 + rng.standard_normal((n, 10)) * 3

        def cv_r2(alpha, X=X, Y=Y):
            return _reference_cv_score(
                X,
                Y,
                alpha,
                n_folds=4,
                criterion="r2"
            )

        chosen = B.select_alpha(
            X,
            Y,
            _DEFAULT_ALPHA_GRID,
            4,
            criterion="r2"
        )
        best = max(_DEFAULT_ALPHA_GRID, key=cv_r2)
        assert cv_r2(chosen) == pytest.approx(cv_r2(best), rel=1e-9)


def test_select_alpha_handles_dual_form():
    """n < p takes the dual path and still matches the reference."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 200))
    Y = rng.standard_normal((30, 6))
    assert B.select_alpha(
        X,
        Y,
        _DEFAULT_ALPHA_GRID,
        3
    ) == _reference_select_alpha(
        X,
        Y,
        _DEFAULT_ALPHA_GRID,
        3,
        criterion="r2"
    )


def test_fit_ridge_matches_closed_form():
    """fit_ridge reproduces the explicit normal-equation solution."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 10))
    Y = rng.standard_normal((50, 3))
    alpha = 2.5
    expected = np.linalg.solve(X.T @ X + alpha * np.eye(10), X.T @ Y)
    np.testing.assert_allclose(B.fit_ridge(X, Y, alpha), expected, rtol=1e-10)


@pytest.mark.skipif(not B.gpu_available(), reason="no CUDA device")
def test_cuda_and_cpu_paths_agree():
    """Forcing each device yields the same selected alpha."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        n = int(rng.integers(40, 120))
        p = int(rng.integers(10, 200))
        X = rng.standard_normal((n, p))
        Y = rng.standard_normal((n, int(rng.integers(4, 40))))
        assert B.select_alpha(
            X,
            Y,
            _DEFAULT_ALPHA_GRID,
            4,
            device="cuda"
        ) == B.select_alpha(
            X,
            Y,
            _DEFAULT_ALPHA_GRID,
            4,
            device="cpu"
        )


def test_small_problems_stay_on_cpu():
    """The size heuristic keeps tiny problems off the GPU, where transfer dominates."""
    small = np.zeros((10, 5))
    big = np.zeros((500, 500))
    assert not B._gpu_worth_it(small)
    assert B._gpu_worth_it(big)


def test_device_override(monkeypatch):
    """BRAIN_CODE_DEVICE forces the device string."""
    monkeypatch.setenv("BRAIN_CODE_DEVICE", "cpu")
    assert B.resolve_device() == "cpu"
    assert not B.gpu_available()


def test_missing_torch_falls_back_to_cpu(monkeypatch):
    """Without torch importable, the backend still resolves and computes."""
    monkeypatch.delenv("BRAIN_CODE_DEVICE", raising=False)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    assert B.resolve_device() == "cpu"
    rng = np.random.default_rng(1)
    X, Y = rng.standard_normal((40, 8)), rng.standard_normal((40, 3))
    assert np.isfinite(B.select_alpha(
        X,
        Y,
        _DEFAULT_ALPHA_GRID,
        4
    ))


@pytest.mark.parametrize("shape", [(50, 120), (120, 50), (64, 64)])
def test_spectrum_matches_svd(shape):
    """Singular values from the Gram eigendecomposition match a direct SVD."""
    rng = np.random.default_rng(sum(shape))
    X = rng.standard_normal(shape)
    reference = np.linalg.svd(X, compute_uv=False)
    got = spectrum(X)
    assert got.shape == reference.shape
    np.testing.assert_allclose(
        got,
        reference,
        rtol=1e-8,
        atol=1e-10
    )


def test_spectrum_is_descending():
    """Singular values come back in descending order."""
    rng = np.random.default_rng(5)
    sv = spectrum(rng.standard_normal((40, 90)))
    assert np.all(np.diff(sv) <= 1e-12)


def test_geometry_edge_cases():
    """All-zero input is NaN; a rank-1 matrix is maximally anisotropic."""
    zeros = np.zeros((10, 5))
    assert np.isnan(effective_rank(zeros))
    assert np.isnan(anisotropy(zeros))

    rng = np.random.default_rng(2)
    rank1 = np.tile(rng.standard_normal((1, 20)), (15, 1))
    assert effective_rank(rank1) == pytest.approx(1.0, abs=1e-6)
    assert anisotropy(rank1) == pytest.approx(1.0, abs=1e-6)


def test_ridge_resolve_device_maps_mps_to_cpu(monkeypatch):
    """An mps request resolves to cpu for the ridge.

    The float64 ridge path uses eigh, which mps supports for neither float64 nor
    the op itself.
    """
    monkeypatch.setenv("BRAIN_CODE_DEVICE", "mps")
    assert B.resolve_device() == "cpu"


def test_ridge_resolve_device_honours_explicit_cuda(monkeypatch):
    monkeypatch.setenv("BRAIN_CODE_DEVICE", "cuda:1")
    assert B.resolve_device() == "cuda:1"


def test_model_device_accepts_mps(monkeypatch):
    """mps is honoured for the model, both as an explicit override and via BRAIN_CODE_DEVICE.

    The model forward runs in fp32, which mps supports.
    """
    monkeypatch.delenv("BRAIN_CODE_DEVICE", raising=False)
    assert B.resolve_model_device("mps") == "mps"
    monkeypatch.setenv("BRAIN_CODE_DEVICE", "mps")
    assert B.resolve_model_device() == "mps"


def test_model_device_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("BRAIN_CODE_DEVICE", "cuda")
    assert B.resolve_model_device("cpu") == "cpu"
