"""Leakage-safe ridge encoding and out-of-fold predictive score contrasts.

This module deliberately fails closed for inferential analyses. Group labels are
required for both inner and outer cross-validation, every observation must
receive exactly one out-of-fold prediction, and all centering/scaling is fit on
training data only.

The quantities produced by :func:`variance_partition_oof` are predictive score
contrasts after refitting. They are not an algebraic decomposition of neural
variance and are therefore allowed to be negative and need not sum to the full
model score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from brain_code.experiments.fold_ci import fold_ci95

# capped at 1e3 like the official ridge; larger alphas let per-target R²
# selection collapse high-dimensional fits to the intercept
_DEFAULT_ALPHA_GRID = np.logspace(
    -3,
    3,
    7,
    dtype=np.float64
)
AlphaStrategy = Literal["global", "per_target"]
SelectionMetric = Literal["r2", "pearson_r"]


@dataclass(frozen=True)
class FoldPreprocessor:
    feature_means: NDArray[np.float64]
    feature_scales: NDArray[np.float64]
    zero_variance_mask: NDArray[np.bool_]
    target_means: NDArray[np.float64]

    def transform_x(self, x: NDArray) -> NDArray[np.float64]:
        x64 = np.asarray(x, dtype=np.float64)
        if x64.ndim != 2 or x64.shape[1] != len(self.feature_means):
            raise ValueError(
                f"Feature shape mismatch: expected (*, {len(self.feature_means)}), "
                f"got {x64.shape}."
            )
        out = (x64 - self.feature_means) / self.feature_scales
        out[:, self.zero_variance_mask] = 0.0
        if not np.isfinite(out).all():
            raise ValueError("Transformed features contain NaN or infinity.")
        return out


def _as_2d(y: NDArray) -> NDArray[np.float64]:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Y must be one- or two-dimensional; got shape {arr.shape}.")
    return arr


def _validate_xy(x: NDArray, y: NDArray, *, label: str) -> tuple[NDArray, NDArray]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = _as_2d(y)
    if x64.ndim != 2:
        raise ValueError(f"{label} X must be two-dimensional; got shape {x64.shape}.")
    if len(x64) != len(y64):
        raise ValueError(f"{label} X/Y row mismatch: {len(x64)} != {len(y64)}.")
    if len(x64) == 0:
        raise ValueError(f"{label} split is empty.")
    if not np.isfinite(x64).all():
        raise ValueError(f"{label} X contains NaN or infinity.")
    if not np.isfinite(y64).all():
        raise ValueError(f"{label} Y contains NaN or infinity.")
    return x64, y64


def _fit_ridge_raw(x: NDArray, y: NDArray, alpha: float) -> NDArray[np.float64]:
    """Closed-form ridge on already centred/scaled arrays, without intercept."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = _as_2d(y)
    n, p = x64.shape
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite and non-negative; got {alpha!r}.")
    if n >= p:
        gram = x64.T @ x64
        gram.flat[:: p + 1] += alpha
        return np.linalg.solve(gram, x64.T @ y64)
    gram = x64 @ x64.T
    gram.flat[:: n + 1] += alpha
    return x64.T @ np.linalg.solve(gram, y64)


# short alias used by the tests
_fit_ridge = _fit_ridge_raw


def _fit_multi_alpha(x: NDArray, y: NDArray, alphas: NDArray) -> NDArray[np.float64]:
    y64 = _as_2d(y)
    alpha_vec = np.asarray(alphas, dtype=np.float64).reshape(-1)
    if len(alpha_vec) != y64.shape[1]:
        raise ValueError("One selected alpha is required per target.")
    coef = np.empty((x.shape[1], y64.shape[1]), dtype=np.float64)
    for alpha in np.unique(alpha_vec):
        cols = np.flatnonzero(alpha_vec == alpha)
        coef[:, cols] = _fit_ridge_raw(x, y64[:, cols], float(alpha))
    return coef


def _fit_preprocessor(x_train: NDArray, y_train: NDArray) -> tuple[FoldPreprocessor, NDArray, NDArray]:
    x64, y64 = _validate_xy(x_train, y_train, label="training")
    means = x64.mean(axis=0)
    raw_scales = x64.std(axis=0, ddof=0)
    zero_var = raw_scales <= np.finfo(np.float64).eps * 100
    scales = raw_scales.copy()
    scales[zero_var] = 1.0
    x_scaled = (x64 - means) / scales
    x_scaled[:, zero_var] = 0.0
    target_means = y64.mean(axis=0)
    y_centered = y64 - target_means
    pre = FoldPreprocessor(
        means,
        scales,
        zero_var,
        target_means
    )
    return pre, x_scaled, y_centered


def _transform_x(pre: FoldPreprocessor, x: NDArray) -> NDArray[np.float64]:
    return pre.transform_x(x)


def _center_and_scale(x_train: NDArray, x_test: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Centers and scales train and test features with train statistics.

    Returns:
        Scaled train and test features, the train means, and the train standard
        deviations with zero-variance features reported as 0.
    """
    x64 = np.asarray(x_train, dtype=np.float64)
    means = x64.mean(axis=0)
    raw = x64.std(axis=0, ddof=0)
    zero = raw <= np.finfo(np.float64).eps * 100
    scales = raw.copy()
    scales[zero] = 1.0
    train = (x64 - means) / scales
    test = (np.asarray(x_test, dtype=np.float64) - means) / scales
    train[:, zero] = 0.0
    test[:, zero] = 0.0
    reported = scales.copy()
    reported[zero] = 0.0
    return (
        train,
        test,
        means,
        reported
    )


def _center_targets(y_train: NDArray, y_test: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    ytr = _as_2d(y_train)
    yte = _as_2d(y_test)
    means = ytr.mean(axis=0)
    return ytr - means, yte - means, means


def _pearson_r_per_col(y_true: NDArray, y_pred: NDArray) -> NDArray[np.float64]:
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Score shape mismatch: {yt.shape} != {yp.shape}.")
    yt0 = yt - yt.mean(axis=0)
    yp0 = yp - yp.mean(axis=0)
    numerator = np.sum(yt0 * yp0, axis=0)
    denominator = np.sqrt(np.sum(yt0**2, axis=0) * np.sum(yp0**2, axis=0))
    out = np.full(yt.shape[1], np.nan, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).eps
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _r2_per_col(y_true: NDArray, y_pred: NDArray) -> NDArray[np.float64]:
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Score shape mismatch: {yt.shape} != {yp.shape}.")
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    out = np.full(yt.shape[1], np.nan, dtype=np.float64)
    valid = ss_tot > np.finfo(np.float64).eps
    out[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return out


def _mean_finite(values: NDArray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("Cannot aggregate an array with no finite scores.")
    return float(finite.mean())


def _selection_scores(
    y_true: NDArray,
    y_pred: NDArray,
    selection_metric: SelectionMetric
) -> NDArray[np.float64]:
    if selection_metric == "r2":
        return _r2_per_col(y_true, y_pred)
    if selection_metric == "pearson_r":
        return _pearson_r_per_col(y_true, y_pred)
    raise ValueError("selection_metric must be 'r2' or 'pearson_r'.")


def _build_inner_groups(groups: NDArray, n_inner_folds: int) -> list[tuple[NDArray, NDArray]]:
    groups_arr = np.asarray(groups)
    if groups_arr.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    if n_inner_folds < 2:
        raise ValueError("n_inner_folds must be at least 2.")
    unique, counts = np.unique(groups_arr, return_counts=True)
    if len(unique) < n_inner_folds:
        raise ValueError(
            f"Cannot build {n_inner_folds} inner folds from {len(unique)} unique groups."
        )
    order = sorted(zip(unique.tolist(), counts.tolist()), key=lambda pair: (-pair[1], str(pair[0])))
    bins: list[list[object]] = [[] for _ in range(n_inner_folds)]
    sizes = np.zeros(n_inner_folds, dtype=np.int64)
    for group, count in order:
        destination = int(np.argmin(sizes))
        bins[destination].append(group)
        sizes[destination] += count
    folds: list[tuple[NDArray, NDArray]] = []
    for held_out in bins:
        val_mask = np.isin(groups_arr, held_out)
        tr_idx = np.flatnonzero(~val_mask)
        va_idx = np.flatnonzero(val_mask)
        if not len(tr_idx) or not len(va_idx):
            raise ValueError("Inner CV produced an empty train or validation split.")
        if set(groups_arr[tr_idx]) & set(groups_arr[va_idx]):
            raise AssertionError("Internal error: group leakage in inner CV.")
        folds.append((tr_idx, va_idx))
    return folds


def _select_alpha_group_aware(
    x: NDArray,
    y: NDArray,
    groups: NDArray,
    alpha_grid: NDArray = _DEFAULT_ALPHA_GRID,
    n_inner_folds: int = 4,
    alpha_strategy: AlphaStrategy = "global",
    selection_metric: SelectionMetric = "r2"
) -> NDArray[np.float64]:
    x64, y64 = _validate_xy(x, y, label="inner-CV")
    groups_arr = np.asarray(groups)
    if len(groups_arr) != len(x64):
        raise ValueError("groups length must match training observations.")
    grid = np.asarray(alpha_grid, dtype=np.float64).reshape(-1)
    if not len(grid) or np.any(grid < 0) or not np.isfinite(grid).all():
        raise ValueError("alpha_grid must contain finite non-negative values.")
    if alpha_strategy not in {"global", "per_target"}:
        raise ValueError("alpha_strategy must be 'global' or 'per_target'.")

    folds = _build_inner_groups(groups_arr, n_inner_folds)
    # alpha is selected on complete inner out-of-fold predictions: an unweighted
    # mean of per-fold scores changes the objective when folds differ in size
    fold_payloads: list[tuple[NDArray, NDArray, NDArray, NDArray, FoldPreprocessor]] = []
    for tr_idx, va_idx in folds:
        pre, xtr, ytr = _fit_preprocessor(x64[tr_idx], y64[tr_idx])
        fold_payloads.append((
            tr_idx,
            va_idx,
            xtr,
            ytr,
            pre
        ))

    means = np.full((len(grid), y64.shape[1]), -np.inf, dtype=np.float64)
    for alpha_i, alpha in enumerate(grid):
        inner_oof = np.full(y64.shape, np.nan, dtype=np.float64)
        for _tr_idx, va_idx, xtr, ytr, pre in fold_payloads:
            xva = _transform_x(pre, x64[va_idx])
            coef = _fit_ridge_raw(xtr, ytr, float(alpha))
            pred = xva @ coef + pre.target_means
            inner_oof[va_idx] = pred
        if not np.isfinite(inner_oof).all():
            raise AssertionError("Internal error: inner CV did not produce complete OOF predictions.")
        selected_score = _selection_scores(y64, inner_oof, selection_metric)
        finite = np.isfinite(selected_score)
        means[alpha_i, finite] = selected_score[finite]

    if alpha_strategy == "global":
        aggregate = np.array([
            _mean_finite(row) if np.isfinite(row).any() else -np.inf
            for row in means
        ])
        if not np.isfinite(aggregate).any():
            raise ValueError("No alpha candidate produced a finite validation score.")
        selected = float(grid[int(np.argmax(aggregate))])
        return np.full(y64.shape[1], selected, dtype=np.float64)
    invalid_targets = np.flatnonzero(~np.isfinite(means).any(axis=0))
    if len(invalid_targets):
        raise ValueError(
            "No alpha candidate produced a finite validation score for target "
            f"columns {invalid_targets[:20].tolist()}."
        )
    best = np.argmax(means, axis=0)
    return grid[best].astype(np.float64)


def eval_encoding_model(
    X_train: NDArray,
    Y_train: NDArray,
    X_test: NDArray,
    Y_test: NDArray,
    alpha_grid: NDArray = _DEFAULT_ALPHA_GRID,
    n_inner_folds: int = 4,
    groups_train: NDArray | None = None,
    groups_test: NDArray | None = None,
    alpha_strategy: AlphaStrategy = "global",
    selection_metric: SelectionMetric = "r2",
    allow_ungrouped: bool = False
) -> dict:
    """Fit on one outer-training split and predict one outer-test split.

    ``groups_train`` is mandatory unless ``allow_ungrouped=True`` is explicitly
    requested for a non-inferential synthetic test. There is no implicit row-CV
    fallback.
    """
    xtr, ytr = _validate_xy(X_train, Y_train, label="outer training")
    xte, yte = _validate_xy(X_test, Y_test, label="outer test")
    if xtr.shape[1] != xte.shape[1] or ytr.shape[1] != yte.shape[1]:
        raise ValueError("Train/test feature or target dimensions do not match.")
    if groups_train is None:
        if not allow_ungrouped:
            raise ValueError("groups_train is required; row-wise inner CV is forbidden.")
        groups_train = np.arange(len(xtr))
    gtr = np.asarray(groups_train)
    if len(gtr) != len(xtr):
        raise ValueError("groups_train length must match X_train.")
    if groups_test is None and not allow_ungrouped:
        raise ValueError(
            "groups_test is required for an inferential outer split so leakage can be checked."
        )
    if groups_test is not None:
        gte = np.asarray(groups_test)
        if len(gte) != len(xte):
            raise ValueError("groups_test length must match X_test.")
        overlap = set(gtr.tolist()) & set(gte.tolist())
        if overlap:
            raise ValueError(f"Outer train/test group leakage detected: {sorted(map(str, overlap))[:5]}.")

    selected_alphas = _select_alpha_group_aware(
        xtr,
        ytr,
        gtr,
        alpha_grid,
        n_inner_folds,
        alpha_strategy,
        selection_metric
    )
    pre, xtr_scaled, ytr_centered = _fit_preprocessor(xtr, ytr)
    xte_scaled = _transform_x(pre, xte)
    coef = _fit_multi_alpha(xtr_scaled, ytr_centered, selected_alphas)
    pred = xte_scaled @ coef + pre.target_means
    return {
        "r": _pearson_r_per_col(yte, pred),
        "r2": _r2_per_col(yte, pred),
        "predictions": pred,
        "coefficients": coef,
        "best_alpha": selected_alphas,
        "feature_means": pre.feature_means,
        "feature_stds": np.where(pre.zero_variance_mask, 0.0, pre.feature_scales),
        "target_means": pre.target_means,
        "alpha_strategy": alpha_strategy,
        "selection_metric": selection_metric
    }


def validate_outer_folds(
    folds: Sequence[tuple[NDArray, NDArray]],
    n_obs: int,
    groups: NDArray
) -> None:
    """Checks outer folds are nonempty, leak-free and test every observation exactly once."""
    if not folds:
        raise ValueError("At least one outer fold is required.")
    groups_arr = np.asarray(groups)
    if groups_arr.ndim != 1 or len(groups_arr) != n_obs:
        raise ValueError("groups must be one-dimensional and match n_obs.")
    test_counts = np.zeros(n_obs, dtype=np.int64)
    for fold_i, (train_idx_raw, test_idx_raw) in enumerate(folds):
        train_idx = np.asarray(train_idx_raw, dtype=np.int64)
        test_idx = np.asarray(test_idx_raw, dtype=np.int64)
        if not len(train_idx) or not len(test_idx):
            raise ValueError(f"Outer fold {fold_i} has an empty train or test split.")
        if np.any(train_idx < 0) or np.any(test_idx < 0) or np.any(train_idx >= n_obs) or np.any(test_idx >= n_obs):
            raise ValueError(f"Outer fold {fold_i} contains an out-of-range index.")
        if len(np.unique(train_idx)) != len(train_idx) or len(np.unique(test_idx)) != len(test_idx):
            raise ValueError(f"Outer fold {fold_i} contains duplicate indices.")
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"Outer fold {fold_i} has observation leakage.")
        group_overlap = set(groups_arr[train_idx].tolist()) & set(groups_arr[test_idx].tolist())
        if group_overlap:
            raise ValueError(
                f"Outer fold {fold_i} has group leakage: {sorted(map(str, group_overlap))[:5]}."
            )
        test_counts[test_idx] += 1
    missing = np.flatnonzero(test_counts == 0)
    repeated = np.flatnonzero(test_counts > 1)
    if len(missing) or len(repeated):
        raise ValueError(
            "Outer folds must assign every observation to exactly one test fold; "
            f"missing={len(missing)}, repeated={len(repeated)}."
        )


def eval_encoding_model_oof(
    X: NDArray,
    Y: NDArray,
    folds: Sequence[tuple[NDArray, NDArray]],
    alpha_grid: NDArray = _DEFAULT_ALPHA_GRID,
    n_inner_folds: int = 4,
    groups: NDArray | None = None,
    alpha_strategy: AlphaStrategy = "global",
    selection_metric: SelectionMetric = "r2"
) -> dict:
    """Fits and scores a ridge encoder across grouped outer folds.

    Returns:
        Per-neuroid scores averaged over folds (``oof_r``, ``oof_r2``), their
        means, per-fold summaries and the out-of-fold predictions.
    """
    x64, y64 = _validate_xy(X, Y, label="full dataset")
    if groups is None:
        raise ValueError("groups is required for out-of-fold inferential evaluation.")
    groups_arr = np.asarray(groups)
    validate_outer_folds(folds, len(x64), groups_arr)

    predictions = np.full(y64.shape, np.nan, dtype=np.float64)
    fold_alphas: list[list[float]] = []
    fold_r_means: list[float] = []
    fold_r2_means: list[float] = []
    fold_r_cols: list[NDArray] = []
    fold_r2_cols: list[NDArray] = []
    for train_idx_raw, test_idx_raw in folds:
        train_idx = np.asarray(train_idx_raw, dtype=np.int64)
        test_idx = np.asarray(test_idx_raw, dtype=np.int64)
        result = eval_encoding_model(
            x64[train_idx],
            y64[train_idx],
            x64[test_idx],
            y64[test_idx],
            alpha_grid=alpha_grid,
            n_inner_folds=n_inner_folds,
            groups_train=groups_arr[train_idx],
            groups_test=groups_arr[test_idx],
            alpha_strategy=alpha_strategy,
            selection_metric=selection_metric
        )
        predictions[test_idx] = result["predictions"]
        fold_alphas.append(np.asarray(result["best_alpha"]).tolist())
        fold_r_means.append(_mean_finite(result["r"]))
        fold_r2_means.append(_mean_finite(result["r2"]))
        fold_r_cols.append(np.asarray(result["r"], dtype=np.float64))
        fold_r2_cols.append(np.asarray(result["r2"], dtype=np.float64))
    if not np.isfinite(predictions).all():
        raise ValueError("OOF predictions are incomplete or non-finite.")
    # per-neuroid scores are averaged over folds, like the official metric (mean of
    # per-split scores); scoring the concatenated predictions once against the
    # global mean is corrupted by per-story fMRI offsets that differ across folds,
    # which pulls r toward 0 and R² negative, so that value is only a diagnostic
    with np.errstate(invalid="ignore"):
        oof_r = np.nanmean(np.vstack(fold_r_cols), axis=0)
        oof_r2 = np.nanmean(np.vstack(fold_r2_cols), axis=0)
    global_concat_r = _pearson_r_per_col(y64, predictions)
    global_concat_r2 = _r2_per_col(y64, predictions)
    return {
        "oof_r": oof_r,
        "oof_r2": oof_r2,
        "oof_r_mean": _mean_finite(oof_r),
        "oof_r2_mean": _mean_finite(oof_r2),
        "global_concat_oof_r_mean": (
            _mean_finite(global_concat_r) if np.isfinite(global_concat_r).any() else float("nan")
        ),
        "global_concat_oof_r2_mean": (
            _mean_finite(global_concat_r2) if np.isfinite(global_concat_r2).any() else float("nan")
        ),
        "predictions": predictions,
        "fold_alphas": fold_alphas,
        "fold_r_means": fold_r_means,
        "fold_r2_means": fold_r2_means,
        "n_folds": len(folds),
        "alpha_strategy": alpha_strategy,
        "selection_metric": selection_metric,
        "scoring": "per_fold_mean"
    }


def variance_partition(
    X_model: NDArray,
    X_nuisance: NDArray,
    Y_train: NDArray,
    Y_test: NDArray,
    X_model_test: NDArray,
    X_nuisance_test: NDArray,
    alpha_grid: NDArray = _DEFAULT_ALPHA_GRID,
    n_inner_folds: int = 4,
    groups_train: NDArray | None = None,
    groups_test: NDArray | None = None,
    alpha_strategy: AlphaStrategy = "global",
    selection_metric: SelectionMetric = "r2"
) -> pd.DataFrame:
    """Single-fold predictive score contrasts; see :func:`variance_partition_oof`."""
    if groups_train is None:
        raise ValueError("groups_train is required for variance_partition.")
    full = eval_encoding_model(
        np.concatenate([X_model, X_nuisance], axis=1),
        Y_train,
        np.concatenate([X_model_test, X_nuisance_test], axis=1),
        Y_test,
        alpha_grid,
        n_inner_folds,
        groups_train,
        groups_test,
        alpha_strategy,
        selection_metric
    )
    model = eval_encoding_model(
        X_model,
        Y_train,
        X_model_test,
        Y_test,
        alpha_grid,
        n_inner_folds,
        groups_train,
        groups_test,
        alpha_strategy,
        selection_metric
    )
    nuisance = eval_encoding_model(
        X_nuisance,
        Y_train,
        X_nuisance_test,
        Y_test,
        alpha_grid,
        n_inner_folds,
        groups_train,
        groups_test,
        alpha_strategy,
        selection_metric
    )
    r2_full = full["r2"]
    r2_model = model["r2"]
    r2_nuisance = nuisance["r2"]
    return pd.DataFrame({
        "r2_full": r2_full,
        "r2_model_only": r2_model,
        "r2_nuisance_only": r2_nuisance,
        "drop_nuisance_delta_r2": r2_full - r2_model,
        "drop_model_delta_r2": r2_full - r2_nuisance,
        # named after variance components for existing callers, but these are
        # score contrasts, not literal unique/shared variance
        "unique_model": r2_full - r2_nuisance,
        "unique_nuisance": r2_full - r2_model,
        "shared": r2_model + r2_nuisance - r2_full,
        "unexplained": 1.0 - r2_full
    })


def variance_partition_oof(
    X_model: NDArray,
    X_nuisance: NDArray,
    Y: NDArray,
    folds: Sequence[tuple[NDArray, NDArray]],
    groups: NDArray,
    alpha_grid: NDArray = _DEFAULT_ALPHA_GRID,
    n_inner_folds: int = 4,
    alpha_strategy: AlphaStrategy = "global",
    selection_metric: SelectionMetric = "r2"
) -> dict:
    """Compute refitted predictive score contrasts from complete OOF predictions."""
    full = eval_encoding_model_oof(
        np.concatenate([X_model, X_nuisance], axis=1),
        Y,
        folds,
        alpha_grid,
        n_inner_folds,
        groups,
        alpha_strategy,
        selection_metric
    )
    model = eval_encoding_model_oof(
        X_model,
        Y,
        folds,
        alpha_grid,
        n_inner_folds,
        groups,
        alpha_strategy,
        selection_metric
    )
    nuisance = eval_encoding_model_oof(
        X_nuisance,
        Y,
        folds,
        alpha_grid,
        n_inner_folds,
        groups,
        alpha_strategy,
        selection_metric
    )
    per_target = pd.DataFrame({
        "oof_r_full": full["oof_r"],
        "oof_r_model_only": model["oof_r"],
        "oof_r_nuisance_only": nuisance["oof_r"],
        "oof_r2_full": full["oof_r2"],
        "oof_r2_model_only": model["oof_r2"],
        "oof_r2_nuisance_only": nuisance["oof_r2"],
        "drop_model_predictive_delta_r2": full["oof_r2"] - nuisance["oof_r2"],
        "drop_nuisance_predictive_delta_r2": full["oof_r2"] - model["oof_r2"]
    })
    # full, model and nuisance share one fold partition, so each per-fold
    # contrast is the elementwise difference of their per-fold scores
    fr2 = np.asarray(full["fold_r2_means"], dtype=np.float64)
    mr2 = np.asarray(model["fold_r2_means"], dtype=np.float64)
    nr2 = np.asarray(nuisance["fold_r2_means"], dtype=np.float64)
    summary_ci = {
        "oof_r_full": fold_ci95(full["fold_r_means"]),
        "oof_r2_full": fold_ci95(fr2),
        "oof_r2_model_only": fold_ci95(mr2),
        "oof_r2_nuisance_only": fold_ci95(nr2),
        "drop_model_predictive_delta_r2": fold_ci95(fr2 - nr2),
        "drop_nuisance_predictive_delta_r2": fold_ci95(fr2 - mr2)
    }
    return {
        "per_target": per_target,
        "summary": {column: _mean_finite(per_target[column].to_numpy()) for column in per_target},
        "summary_ci": summary_ci,
        "models": {
            "full": full,
            "model_only": model,
            "nuisance_only": nuisance
        },
        "interpretation": "refitted_predictive_score_contrasts_not_variance_components"
    }
