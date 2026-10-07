"""Manifold-geometry pressure metrics and causal PC ablation.

Two questions, one module:

**G1 (descriptive).** What shape does a training corpus impose on a model's
representation manifold?  ``spectral_profile`` summarises one layer's
activation matrix with four scale-invariant numbers read off the covariance
spectrum: effective rank, anisotropy, participation ratio, and the power-law
decay exponent.  Sweeping this over layers and corpora gives the geometry side
of a geometry-vs-alignment correlation.

**G2 (causal).** Is alignment *carried* by the manifold's dominant directions?
``pc_ablation_curve`` removes the top-``m`` principal directions and re-scores,
so a corpus whose brain-predictive signal lives in a few leading directions
separates from one where it is spread thin.

``effective_rank`` and ``anisotropy`` are re-used from :mod:`geometry`; the two
metrics added here (participation ratio, spectral decay) are the ones that
distinguish *how* variance is distributed rather than merely *how much* is
concentrated in the top direction.

The projector in G2 is fit **inside each fold** on training rows only.  PCA is
unsupervised, so leakage would be mild, but a projector fit on all rows still
sees held-out covariance structure, and this repository treats test-set-derived
analysis choices as a validity defect (cf. the absent ``nested_cv`` mode in
``brain_score_harness``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from brain_code.experiments.geometry import anisotropy, effective_rank


def covariance_spectrum(X: NDArray) -> NDArray:
    """Eigenvalues of the column covariance of ``X``, descending (float64).

    Computed from the singular values of the mean-centred matrix rather than by
    forming the covariance explicitly: squaring the Gram matrix would square the
    condition number, and these spectra span several orders of magnitude.
    """
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n = max(len(Xc) - 1, 1)
    sv = np.linalg.svd(Xc, compute_uv=False)
    return (sv * sv) / n


def participation_ratio(X: NDArray) -> float:
    """``(sum lambda)^2 / sum(lambda^2)`` over the covariance spectrum.

    The standard manifold-dimensionality measure: 1 for a line, ``d`` for an
    isotropic cloud in ``d`` dimensions.  Unlike effective rank (an entropy,
    which weights the long tail heavily) the participation ratio is dominated by
    the largest eigenvalues, so the two disagree exactly when a representation
    has a sharp head and a fat tail -- which is the regime of interest here.

    Scale-invariant.  Returns NaN for an all-zero matrix.
    """
    lam = covariance_spectrum(X)
    denom = float((lam * lam).sum())
    if denom <= 0:
        return float("nan")
    total = float(lam.sum())
    return float(total * total / denom)


def spectral_decay(
    X: NDArray,
    *,
    i_min: int = 1,
    i_max: int | None = None
) -> float:
    """Power-law exponent ``alpha`` of the covariance spectrum, ``lambda_i ~ i**(-alpha)``.

    Fit by ordinary least squares of ``log lambda_i`` on ``log i`` over the
    1-indexed eigenvalue range ``[i_min, i_max]``.  Larger ``alpha`` means
    variance falls off faster with dimension, i.e. a lower-dimensional, more
    compressible manifold.

    Args:
        X: Activation matrix ``(n_samples, n_features)``.
        i_min: First eigenvalue index to fit (1-indexed, inclusive).
        i_max: Last index to fit (inclusive); defaults to the full spectrum.

    Returns:
        float: The fitted exponent, or NaN if fewer than two usable
        eigenvalues fall in the requested range.
    """
    lam = covariance_spectrum(X)
    hi = len(lam) if i_max is None else min(int(i_max), len(lam))
    lo = max(int(i_min), 1)
    if hi - lo + 1 < 2:
        return float("nan")

    idx = np.arange(lo, hi + 1, dtype=np.float64)
    window = lam[lo - 1 : hi]
    # eigenvalues at the numerical noise floor carry no slope information
    keep = window > 0
    if keep.sum() < 2:
        return float("nan")

    slope = np.polyfit(np.log(idx[keep]), np.log(window[keep]), 1)[0]
    return float(-slope)


def spectral_profile(X: NDArray, **decay_kwargs) -> dict[str, float]:
    """All four geometry metrics for one activation matrix.

    ``effective_rank`` and ``anisotropy`` are computed on the mean-centred
    matrix so that every number in the profile describes the same covariance
    spectrum; :mod:`geometry` deliberately does not centre, because its callers
    pass already-centred inputs.
    """
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n_samples, n_features = Xc.shape
    return {
        "effective_rank": effective_rank(Xc),
        "anisotropy": anisotropy(Xc),
        "participation_ratio": participation_ratio(Xc),
        "spectral_decay": spectral_decay(Xc, **decay_kwargs),
        "n_samples": int(n_samples),
        "n_features": int(n_features)
    }


@dataclass(frozen=True)
class PCRemover:
    """Fixed affine map that centres and projects out ``m`` principal directions.

    Frozen and stateless on purpose: the same map must apply unchanged to
    held-out rows, so nothing may be re-estimated at transform time.
    """

    mean: NDArray
    projector: NDArray
    n_removed: int

    def transform(self, X: NDArray) -> NDArray:
        return (np.asarray(X, dtype=np.float64) - self.mean) @ self.projector


def fit_pc_remover(X: NDArray, m: int) -> PCRemover:
    """Fit the ``I - V V^T`` projector that deletes ``X``'s top-``m`` PCs.

    Args:
        X: Training rows only ``(n_samples, n_features)``.
        m: Number of leading principal directions to remove; 0 leaves a
            centring-only map.
    """
    Xm = np.asarray(X, dtype=np.float64)
    mean = Xm.mean(axis=0)
    d = Xm.shape[1]
    m = int(m)
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    if m > d:
        raise ValueError(f"cannot remove {m} directions from {d}-dimensional data")

    projector = np.eye(d, dtype=np.float64)
    if m > 0:
        _, _, vt = np.linalg.svd(Xm - mean, full_matrices=False)
        v = vt[:m].T
        projector = projector - v @ v.T
    return PCRemover(mean=mean, projector=projector, n_removed=m)


def _pearson_per_column(Y: NDArray, Yhat: NDArray) -> NDArray:
    """Pearson r for each target column; NaN where a column is constant."""
    yc = Y - Y.mean(axis=0, keepdims=True)
    pc = Yhat - Yhat.mean(axis=0, keepdims=True)
    denom = np.sqrt((yc * yc).sum(axis=0) * (pc * pc).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (yc * pc).sum(axis=0) / denom
    return np.where(denom > 0, r, np.nan)


def grouped_ridge_fold_scorer(
    alpha_grid=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5),
    n_inner_folds: int = 4,
    *,
    groups: NDArray,
    alpha_strategy: str = "per_target"
):
    """Fold scorer with exact parity to ``subspace_erasure._score_condition``.

    Delegates to :func:`variance_partitioning.eval_encoding_model`, the same
    per-fold routine ``eval_encoding_model_oof`` calls, with grouped inner CV and
    per-target alpha selection.  Using anything else here would give the ablation
    curve a different ``m=0`` baseline than the layer sweep it is reported
    against, and every retained-fraction would be wrong.

    Args:
        groups: Grouping coordinate for **all** rows; sliced per fold.
    """
    from brain_code.experiments.variance_partitioning import eval_encoding_model

    grid = np.asarray(list(alpha_grid), dtype=np.float64)
    groups_all = np.asarray(groups)

    def scorer(
        X_tr,
        Y_tr,
        X_te,
        Y_te,
        *,
        train_idx,
        test_idx
    ):
        return eval_encoding_model(
            X_tr,
            Y_tr,
            X_te,
            Y_te,
            alpha_grid=grid,
            n_inner_folds=n_inner_folds,
            groups_train=groups_all[train_idx],
            groups_test=groups_all[test_idx],
            alpha_strategy=alpha_strategy
        )["predictions"]

    return scorer


def pc_ablation_curve(
    X: NDArray,
    Y: NDArray,
    folds: list[tuple[NDArray, NDArray]],
    m_grid: list[int],
    *,
    fold_scorer
) -> list[dict]:
    """Alignment retained after deleting the top-``m`` principal directions.

    For every ``m`` the projector is re-fit on each fold's training rows, applied
    to both sides of the split, and the resulting out-of-fold predictions are
    scored as one pooled matrix -- not averaged per fold, matching how
    ``subspace_erasure._score_condition`` scores complete OOF predictions.

    Args:
        X: Model activations ``(n_stimuli, n_features)``.
        Y: Neural responses ``(n_stimuli, n_neuroids)``.
        folds: ``(train_idx, test_idx)`` index pairs covering every row once.
        m_grid: Numbers of leading directions to remove.
        fold_scorer: ``(X_tr, Y_tr, X_te, Y_te) -> predictions for X_te``.

    Returns:
        list[dict]: One row per ``m`` with ``score`` (mean Pearson r over
        neuroids) and ``variance_removed`` (fraction of training activation
        variance the projector deleted, averaged over folds).
    """
    Xa = np.asarray(X, dtype=np.float64)
    Ya = np.asarray(Y, dtype=np.float64)
    if Ya.ndim == 1:
        Ya = Ya[:, None]

    curve = []
    for m in m_grid:
        fold_r, removed = [], []
        for train_idx, test_idx in folds:
            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)

            remover = fit_pc_remover(Xa[train_idx], m)
            x_tr = remover.transform(Xa[train_idx])
            x_te = remover.transform(Xa[test_idx])

            before = float(((Xa[train_idx] - remover.mean) ** 2).sum())
            after = float((x_tr**2).sum())
            removed.append(0.0 if before <= 0 else 1.0 - after / before)

            pred = np.asarray(
                fold_scorer(
                    x_tr,
                    Ya[train_idx],
                    x_te,
                    Ya[test_idx],
                    train_idx=train_idx,
                    test_idx=test_idx
                ),
                dtype=np.float64
            )
            fold_r.append(_pearson_per_column(Ya[test_idx], pred))

        # average each neuroid over folds before the mean over neuroids, as
        # eval_encoding_model_oof does: scoring concatenated OOF predictions once
        # against the global mean is corrupted by per-story fMRI offsets
        with np.errstate(invalid="ignore"):
            per_neuroid = np.nanmean(np.vstack(fold_r), axis=0)
            score = float(np.nanmean(per_neuroid)) if np.isfinite(per_neuroid).any() else float("nan")

        curve.append({
            "m": int(m),
            "score": score,
            "variance_removed": float(np.mean(removed))
        })
    return curve


def ridge_fold_scorer(
    alpha_grid=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5),
    n_inner_folds: int = 4,
    device: str | None = None
):
    """Build the ``fold_scorer`` that :func:`pc_ablation_curve` needs.

    Selects one ridge alpha per fold by inner k-fold CV on the training rows,
    then refits on the full training block.  Features and targets are centred on
    training statistics and the target mean is added back, which is equivalent
    to fitting an unpenalised intercept.

    ``Y_te`` is accepted only so the callable matches the ``fold_scorer``
    signature; it is never read.
    """
    from brain_code.experiments.linalg_backend import fit_ridge, select_alpha

    grid = np.asarray(list(alpha_grid), dtype=np.float64)

    def scorer(
        X_tr,
        Y_tr,
        X_te,
        _Y_te,
        **_
    ):
        x_tr = np.asarray(X_tr, dtype=np.float64)
        y_tr = np.asarray(Y_tr, dtype=np.float64)
        x_te = np.asarray(X_te, dtype=np.float64)

        x_mu, y_mu = x_tr.mean(axis=0), y_tr.mean(axis=0)
        xc, yc = x_tr - x_mu, y_tr - y_mu

        alpha = float(grid[0]) if grid.size == 1 else float(select_alpha(
            xc,
            yc,
            grid,
            n_inner_folds,
            device=device
        ))
        beta = fit_ridge(xc, yc, alpha)
        return (x_te - x_mu) @ beta + y_mu

    return scorer


#: Profile fields that are not geometry metrics.  ``alignment_r`` is here
#: because the runner writes each layer's score back into its profile row for
#: reporting; correlating it would just recover rho=1 against itself.
_NON_METRIC_FIELDS = frozenset({"layer", "n_samples", "n_features", "alignment_r"})


def layerwise_spectral_profile(reps_by_layer: dict[int, NDArray], **decay_kwargs) -> list[dict]:
    """Spectral profile of every layer, ordered by layer index.

    Args:
        reps_by_layer: Layer index -> activation matrix over the *same* stimuli,
            so profiles are comparable across layers and corpora.
    """
    return [
        {"layer": int(layer), **spectral_profile(reps_by_layer[layer], **decay_kwargs)}
        for layer in sorted(reps_by_layer, key=int)
    ]


def geometry_alignment_correlation(profiles: list[dict], scores: list[float]) -> dict[str, dict]:
    """Rank-correlate each geometry metric against layer-wise alignment.

    Spearman rather than Pearson: the question is whether a geometric pressure
    tracks alignment monotonically across depth, not whether the relationship is
    linear, and effective rank and participation ratio live on very different
    scales.

    Returns:
        dict: metric name -> ``{"rho": float, "n": int}``.
    """
    from scipy.stats import spearmanr

    if len(profiles) != len(scores):
        raise ValueError(
            f"profiles and scores must have equal length; "
            f"got {len(profiles)} and {len(scores)}"
        )

    y = np.asarray(scores, dtype=np.float64)
    metrics = [k for k in profiles[0] if k not in _NON_METRIC_FIELDS]

    out: dict[str, dict] = {}
    for metric in metrics:
        x = np.asarray([p[metric] for p in profiles], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            out[metric] = {"rho": float("nan"), "n": int(finite.sum())}
            continue
        rho = spearmanr(x[finite], y[finite]).statistic
        out[metric] = {"rho": float(rho), "n": int(finite.sum())}
    return out
