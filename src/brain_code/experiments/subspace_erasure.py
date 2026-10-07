"""Cross-fitted LEACE concept erasure for encoding analysis.

Uses **LEACE** (Least-squares Concept Erasure, Belrose et al. 2023, "LEACE:
Perfect linear concept erasure in closed form", NeurIPS), which computes, in
closed form, the affine map that removes ALL linearly-available information about
a concept while perturbing the representation as little as possible (minimum
expected squared distance). Iterative probe projection such as INLP (Ravfogel et
al. 2020) removes only the directions its probes find, with no such guarantee.

Protocol (cross-fitted, leak-free):
    1. On a PROBE DATASET (NOT the brain stimuli), split into train/test.
    2. Fit the LEACE eraser on the probe TRAIN split only; freeze it.
    3. Verify erasure on the probe TEST split with a *fresh* linear probe
       (should fall to chance) and a *fresh* non-linear probe (may retain
       signal — LEACE only guarantees LINEAR erasure).
    4. Check SELECTIVITY: unrelated attributes must remain decodable.
    5. Apply the FROZEN eraser to brain-stimulus representations and compare
       brain alignment (ridge encoding R²) before vs. after erasure, using
       grouped cross-validation so no passage/story straddles train/test.
    6. Compare the erasure-induced loss against a family of MATCHED CONTROL
       projections (random + PCA, matched on removed rank / variance / norm /
       effective-rank change) to form a null distribution.

The LEACE guarantee (stated precisely):
    For X drawn from the fit distribution, ``Cov(erase(X), Z) = 0`` exactly (up
    to float64 rounding).  Equivalently, no linear map can predict the concept
    Z from ``erase(X)`` better than a constant.  This is *linear* erasure only.

Interpretation constraint (see also :func:`representation_effect_gate`):
    A drop in brain score after erasure means the erased linear subspace
    contributed to the *linear* encoder — NOT that the brain represents the
    concept causally in the same coordinates, and NOT that all concept information
    was removed.  This is evidence of a representation-level correlate.
    Do not claim causal necessity.  Report an association: the linear subspace
    correlates with brain-predictive variance.

References:
    Belrose et al. (2023) — LEACE (least-squares concept erasure, closed form)
    Ravfogel et al. (2020) — INLP (iterative nullspace projection)
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from brain_code.experiments.geometry import (
    anisotropy,
    effective_rank,
    total_variance
)
from brain_code.experiments.probes import (
    chance_level,
    linear_probe_accuracy,
    nonlinear_probe_accuracy,
    selectivity_report
)
from brain_code.experiments.split_manifests import build_grouped_folds
from brain_code.experiments.variance_partitioning import (
    _DEFAULT_ALPHA_GRID,
    eval_encoding_model_oof
)

CONTROL_FAMILIES = (
    "random_rank",
    "pca_rank",
    "random_var",
    "pca_var",
    "random_norm",
    "pca_norm",
    "pca_effrank"
)


class LeaceEraser:
    """Least-squares concept eraser (Belrose et al. 2023), numpy, float64.

    Fits the closed-form affine map ``r(x) = (x - mu) @ M.T + mu`` that removes
    all linearly-decodable information about the concept ``y`` while minimising
    the expected squared perturbation.  Fit on one dataset, applied frozen to
    others.

    Args:
        eig_floor_rtol: Eigenvalues of the covariance below
            ``eig_floor_rtol * max_eig`` are treated as zero when forming the
            (pseudo-)inverse whitening — handles rank-deficient covariance.
        svd_rtol: Singular values of the whitened cross-covariance below
            ``svd_rtol * max_sv`` are treated as zero when determining the
            removed rank.
    """

    def __init__(self, eig_floor_rtol: float = 1e-10, svd_rtol: float = 1e-9):
        self.eig_floor_rtol = eig_floor_rtol
        self.svd_rtol = svd_rtol
        self.mu_: NDArray | None = None
        self.M_: NDArray | None = None            # (d, d) float64, column-vector form
        self.removed_rank_: int = 0

    def fit(self, X: NDArray, y: NDArray) -> "LeaceEraser":
        """Fit on ``X`` (n, d) with integer concept labels ``y`` (n,).

        Everything is computed in float64.  A one-hot encoding of ``y`` gives a
        centered target of rank ``k - 1``, so the removed rank is ``k - 1``
        (``1`` for a binary concept).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, d = X.shape

        classes = np.unique(y)
        k = len(classes)
        if k < 2 or n < 3:
            warnings.warn(
                "LeaceEraser.fit: fewer than 2 classes or <3 samples; "
                "eraser is the identity (nothing to erase).",
                stacklevel=2
            )
            self.mu_ = X.mean(axis=0)
            self.M_ = np.eye(d, dtype=np.float64)
            self.removed_rank_ = 0
            return self

        Z = np.eye(k, dtype=np.float64)[np.searchsorted(classes, y)]

        mu = X.mean(axis=0)
        Xc = X - mu
        Zc = Z - Z.mean(axis=0)

        Sxx = Xc.T @ Xc / n            # (d, d) symmetric PSD
        Sxz = Xc.T @ Zc / n            # (d, k)

        evals, evecs = np.linalg.eigh(Sxx)
        floor = self.eig_floor_rtol * max(evals.max(), 0.0)
        # filled by mask rather than np.where, which would evaluate evals ** -0.5
        # on the tiny negative round-off eigenvalues and warn
        positive = evals > floor
        inv_sqrt = np.zeros_like(evals)
        sqrt = np.zeros_like(evals)
        inv_sqrt[positive] = evals[positive] ** -0.5
        sqrt[positive] = np.sqrt(evals[positive])
        W = (evecs * inv_sqrt) @ evecs.T        # Sxx^{-1/2}  (symmetric)
        Wpinv = (evecs * sqrt) @ evecs.T        # Sxx^{+1/2}  (pseudo-inverse of W)

        # orthogonal projector onto the column space of W @ Sxz
        WSxz = W @ Sxz
        U, s, _ = np.linalg.svd(WSxz, full_matrices=False)
        if s.size == 0 or s.max() == 0:
            r = 0
        else:
            r = int((s > self.svd_rtol * s.max()).sum())
        Ur = U[:, :r]
        P = Ur @ Ur.T                            # (d, d)

        # oblique eraser in column-vector form; the row-vector transform uses M.T
        self.M_ = np.eye(d) - Wpinv @ P @ W
        self.mu_ = mu
        self.removed_rank_ = r
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Apply the frozen eraser to ``X`` (n, d). Output matches input dtype.

        NOTE the ``.T``: in the row-vector convention (features in rows) the
        linear map is applied as ``x_row @ M.T``.  Using ``M`` instead of
        ``M.T`` silently produces wrong-but-plausible numbers because ``M`` is
        not symmetric.
        """
        assert self.M_ is not None and self.mu_ is not None, "Call fit() first."
        Xd = np.asarray(X, dtype=np.float64)
        out = (Xd - self.mu_) @ self.M_.T + self.mu_
        return out.astype(np.asarray(X).dtype)

    def m_hash(self) -> str:
        """Deterministic sha256 (16 hex) of the eraser matrix — for auditing."""
        assert self.M_ is not None, "Call fit() first."
        return hashlib.sha256(self.M_.astype(np.float64).tobytes()).hexdigest()[:16]


class MatchedControl:
    """A frozen affine control map ``r(x) = (x - mu) @ A + mu`` (A symmetric).

    Unlike LEACE, ``A`` is stored directly in the row-vector convention (it is
    symmetric for every control family, so ``A == A.T``).  ``meta`` records the
    quantity this control was built to match and the value it achieved.
    """

    def __init__(
        self,
        A: NDArray,
        mu: NDArray,
        family: str,
        meta: dict
    ):
        self.A = A
        self.mu = mu
        self.family = family
        self.meta = meta

    def transform(self, X: NDArray) -> NDArray:
        Xd = np.asarray(X, dtype=np.float64)
        out = (Xd - self.mu) @ self.A + self.mu
        return out.astype(np.asarray(X).dtype)


def _random_removal(d: int, m: int, rng: np.random.Generator) -> NDArray:
    """Symmetric projector that removes a random ``m``-dim subspace: I - QQ^T."""
    if m <= 0:
        return np.eye(d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
    Q = Q[:, :m]
    return np.eye(d) - Q @ Q.T


def _pca_removal(evecs_desc: NDArray, m: int) -> NDArray:
    """Symmetric projector that removes the top-``m`` principal directions."""
    d = evecs_desc.shape[0]
    if m <= 0:
        return np.eye(d)
    V = evecs_desc[:, :m]
    return np.eye(d) - V @ V.T


def leace_realized_stats(eraser: LeaceEraser, X_fit: NDArray) -> dict:
    """Quantities LEACE actually removes on its fit data (targets for matching)."""
    Xd = np.asarray(X_fit, dtype=np.float64)
    Ed = np.asarray(eraser.transform(Xd), dtype=np.float64)
    delta = Xd - Ed
    return {
        "removed_rank": int(eraser.removed_rank_),
        "variance_removed": float(total_variance(Xd) - total_variance(Ed)),
        "perturbation_norm": float(np.linalg.norm(delta, axis=1).mean()),
        "effrank_after": float(effective_rank(Ed))
    }


def build_matched_control(
    family: str,
    X_fit: NDArray,
    leace_stats: dict,
    seed: int = 0
) -> MatchedControl:
    """Construct a control projection fit on ``X_fit`` matched to LEACE.

    Each family targets exactly one quantity (removed rank, activation variance,
    perturbation norm, or effective-rank change): the four cannot be matched
    simultaneously by a single projection, which is why we build a family and
    report LEACE relative to the resulting null distribution.
    """
    X_fit = np.asarray(X_fit, dtype=np.float64)
    n, d = X_fit.shape
    mu = X_fit.mean(axis=0)
    Xc = X_fit - mu
    rng = np.random.default_rng(seed)
    r = max(int(leace_stats["removed_rank"]), 0)

    cov = Xc.T @ Xc / max(n, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    def _removed_var(A: NDArray) -> float:
        return float(total_variance(Xc) - total_variance(Xc @ A))

    def _delta_mean_norm(A: NDArray) -> float:
        delta = Xc - Xc @ A
        return float(np.linalg.norm(delta, axis=1).mean())

    def _grow_base(make_base, measure, target: float) -> NDArray:
        """Smallest-``m`` base whose measure at alpha=1 reaches ``target``.

        A random ``r``-dim subspace often cannot hold as much variance/norm as
        LEACE removes (LEACE targets high-variance concept directions), so we
        grow the removed dimension until the target is reachable, then the
        caller rescales by alpha in [0, 1] to hit it exactly.
        """
        base = make_base(d)                       # full removal: an upper bound
        for m in range(1, d + 1):
            b = make_base(m)
            if measure(b) >= target:
                return b
        return base

    if family == "random_rank":
        A = _random_removal(d, r, rng)
    elif family == "pca_rank":
        A = _pca_removal(evecs, r)
    elif family in ("random_var", "pca_var"):
        target = leace_stats["variance_removed"]
        make_base = ((lambda m: _random_removal(d, m, rng)) if family == "random_var"
                     else (lambda m: _pca_removal(evecs, m)))
        base = _grow_base(make_base, _removed_var, target)
        base_var = _removed_var(base)
        # removed_var(alpha) = base_var * (1 - (1-alpha)^2); solve for alpha in [0, 1]
        ratio = 0.0 if base_var <= 0 else min(target / base_var, 1.0)
        alpha = 1.0 - np.sqrt(max(0.0, 1.0 - ratio))
        A = np.eye(d) - alpha * (np.eye(d) - base)
    elif family in ("random_norm", "pca_norm"):
        target = leace_stats["perturbation_norm"]
        make_base = ((lambda m: _random_removal(d, m, rng)) if family == "random_norm"
                     else (lambda m: _pca_removal(evecs, m)))
        base = _grow_base(make_base, _delta_mean_norm, target)
        base_norm = _delta_mean_norm(base)
        # Delta_norm(alpha) = alpha * base_norm; solve for alpha in [0, 1].
        alpha = 0.0 if base_norm <= 0 else min(target / base_norm, 1.0)
        A = np.eye(d) - alpha * (np.eye(d) - base)
    elif family == "pca_effrank":
        target = leace_stats["effrank_after"]
        best_m, best_gap = 1, np.inf
        for m in range(1, d):
            A_m = _pca_removal(evecs, m)
            gap = abs(effective_rank(Xc @ A_m) - target)
            if gap < best_gap:
                best_gap, best_m = gap, m
        A = _pca_removal(evecs, best_m)
    else:
        raise ValueError(f"Unknown control family: {family!r}")

    A_kept = Xc @ A
    meta = {
        "family": family,
        "achieved_rank": int(np.linalg.matrix_rank(A)),
        "achieved_variance_removed": float(total_variance(Xc) - total_variance(A_kept)),
        "achieved_perturbation_norm": _delta_mean_norm(A),
        "achieved_effrank_after": float(effective_rank(A_kept))
    }
    return MatchedControl(
        A=A,
        mu=mu,
        family=family,
        meta=meta
    )


def _score_condition(
    X: NDArray,
    Y: NDArray,
    folds: list[tuple[NDArray, NDArray]],
    alpha_grid: NDArray = _DEFAULT_ALPHA_GRID,
    n_inner_folds: int = 4,
    groups: NDArray | None = None
) -> tuple[NDArray, float, float]:
    """Score complete out-of-fold predictions rather than averaging fold metrics."""
    if groups is None:
        raise ValueError(
            "neural_groups is required; row-wise or ungrouped scoring is forbidden."
        )
    result = eval_encoding_model_oof(
        X,
        Y,
        folds,
        alpha_grid=alpha_grid,
        n_inner_folds=n_inner_folds,
        groups=np.asarray(groups),
        alpha_strategy="per_target"
    )
    return (
        np.asarray(result["oof_r2"], dtype=np.float64),
        float(result["oof_r2_mean"]),
        float(result["oof_r_mean"])
    )


def _folds_from_groups(groups: NDArray, n_folds: int, seed: int) -> list[tuple[NDArray, NDArray]]:
    """Grouped outer folds over neural stimuli (reuses split_manifests)."""
    df = pd.DataFrame({"__group__": np.asarray(groups)})
    return build_grouped_folds(
        df,
        group_col="__group__",
        n_folds=n_folds,
        seed=seed
    )


def cross_fitted_erasure_analysis(
    *,
    probe_reps: NDArray,
    probe_labels: NDArray,
    probe_train_idx: NDArray,
    probe_test_idx: NDArray,
    neural_reps: NDArray,
    neural_responses: NDArray,
    neural_groups: NDArray,
    selectivity_labels: dict[str, NDArray] | None = None,
    probe_groups: NDArray | None = None,
    confirmatory: bool = False,
    n_outer_folds: int = 5,
    n_inner_folds: int = 4,
    n_control_seeds: int = 50,
    control_families: tuple[str, ...] = CONTROL_FAMILIES,
    control_percentile: float = 95.0,
    seed: int = 42,
    alpha_grid: NDArray = _DEFAULT_ALPHA_GRID
) -> dict:
    """Full LEACE erasure analysis on arrays (no HF / brainscore in the core).

    LEACE is fit on the probe TRAIN split and frozen; it is then applied
    unchanged to the probe TEST split (erasure + selectivity checks) and to the
    neural stimuli (brain-score impact).  Matched controls are fit on the SAME
    probe-train reps and applied the same way, so no condition has an
    information advantage.

    Returns a structured dict (see keys below).  ``random_*`` control families
    yield a sampling null over ``n_control_seeds``; ``pca_*`` families are
    deterministic given the fit data, so they are reported as single points.
    """
    probe_reps = np.asarray(probe_reps)
    neural_reps = np.asarray(neural_reps)
    labels = np.asarray(probe_labels)
    neural_groups = np.asarray(neural_groups)
    Y = np.asarray(neural_responses)
    if probe_reps.ndim != 2 or neural_reps.ndim != 2:
        raise ValueError("probe_reps and neural_reps must be two-dimensional.")
    if len(probe_reps) != len(labels):
        raise ValueError("probe_reps and probe_labels length mismatch.")
    if len(neural_reps) != len(Y) or len(neural_reps) != len(neural_groups):
        raise ValueError("Neural representations, responses, and groups must align.")
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2:
        raise ValueError("neural_responses must be one- or two-dimensional.")
    if not all(np.isfinite(value).all() for value in (probe_reps, neural_reps, Y)):
        raise ValueError("Representations and neural responses must be finite.")

    tr = np.asarray(probe_train_idx, dtype=np.int64)
    te = np.asarray(probe_test_idx, dtype=np.int64)
    if tr.ndim != 1 or te.ndim != 1 or not len(tr) or not len(te):
        raise ValueError("Probe train/test indices must be nonempty one-dimensional arrays.")
    if len(np.unique(tr)) != len(tr) or len(np.unique(te)) != len(te):
        raise ValueError("Probe train/test indices contain duplicates.")
    if np.intersect1d(tr, te).size:
        raise ValueError("Probe train and test observations overlap.")
    if np.any(tr < 0) or np.any(te < 0) or np.any(tr >= len(labels)) or np.any(te >= len(labels)):
        raise ValueError("Probe split contains an out-of-range index.")
    if probe_groups is None:
        if confirmatory:
            raise ValueError(
                "confirmatory erasure requires source/template groups for the probe split."
            )
    else:
        probe_groups_arr = np.asarray(probe_groups)
        if len(probe_groups_arr) != len(labels):
            raise ValueError("probe_groups length must match probe labels.")
        overlap = set(probe_groups_arr[tr].tolist()) & set(probe_groups_arr[te].tolist())
        if overlap:
            raise ValueError(
                "Probe source/template groups cross train and test: "
                + ", ".join(sorted(map(str, overlap))[:10])
            )
    if confirmatory and n_control_seeds < 1000:
        raise ValueError(
            "confirmatory erasure requires at least 1,000 random-control draws."
        )
    if confirmatory and not selectivity_labels:
        raise ValueError("confirmatory erasure requires unrelated selectivity labels.")

    Xp_tr, Xp_te = probe_reps[tr], probe_reps[te]
    yp_tr, yp_te = labels[tr], labels[te]
    if len(np.unique(yp_tr)) < 2 or len(np.unique(yp_te)) < 2:
        raise ValueError("Every probe split must contain at least two target classes.")

    # 1. fit and freeze leace on the probe train split
    eraser = LeaceEraser().fit(Xp_tr, yp_tr)
    Ep_tr, Ep_te = eraser.transform(Xp_tr), eraser.transform(Xp_te)

    # 2. erasure check with fresh linear and nonlinear probes on the probe test split
    ch = chance_level(yp_te)
    erasure_check = {
        "target_linear_acc_orig": linear_probe_accuracy(
            Xp_tr,
            yp_tr,
            Xp_te,
            yp_te
        ),
        "target_linear_acc_erased": linear_probe_accuracy(
            Ep_tr,
            yp_tr,
            Ep_te,
            yp_te
        ),
        "target_nonlinear_acc_orig": nonlinear_probe_accuracy(
            Xp_tr,
            yp_tr,
            Xp_te,
            yp_te,
            seed=seed
        ),
        "target_nonlinear_acc_erased": nonlinear_probe_accuracy(
            Ep_tr,
            yp_tr,
            Ep_te,
            yp_te,
            seed=seed
        ),
        "chance_uniform": ch["uniform"],
        "chance_majority": ch["majority"]
    }

    # 3. selectivity: the target must drop linearly while unrelated attributes survive
    selectivity = None
    if selectivity_labels:
        attr_tr = {"target": yp_tr}
        attr_te = {"target": yp_te}
        for name, attr_labels in selectivity_labels.items():
            attr_labels = np.asarray(attr_labels)
            attr_tr[name] = attr_labels[tr]
            attr_te[name] = attr_labels[te]
        selectivity = selectivity_report(
            Xp_tr,
            Xp_te,
            Ep_tr,
            Ep_te,
            attr_tr,
            attr_te,
            target_attribute="target",
            seed=seed
        )

    # 4. brain score before and after erasure, grouped outer cv
    folds = _folds_from_groups(neural_groups, n_outer_folds, seed)
    orig_r2, orig_r2_mean, orig_r_mean = _score_condition(
        neural_reps,
        Y,
        folds,
        alpha_grid,
        n_inner_folds,
        groups=neural_groups
    )
    En = eraser.transform(neural_reps)
    leace_r2, leace_r2_mean, leace_r_mean = _score_condition(
        En,
        Y,
        folds,
        alpha_grid,
        n_inner_folds,
        groups=neural_groups
    )
    raw_loss_r2 = orig_r2 - leace_r2
    leace_loss = orig_r2_mean - leace_r2_mean

    # 5. matched-control null, fit on the probe train split and frozen
    stats = leace_realized_stats(eraser, Xp_tr)
    controls: dict[str, dict] = {}
    for family in control_families:
        if family.startswith("random"):
            losses = []
            for s in range(n_control_seeds):
                ctrl = build_matched_control(
                    family,
                    Xp_tr,
                    stats,
                    seed=seed * 1000 + s
                )
                _, r2m, _ = _score_condition(
                    ctrl.transform(neural_reps),
                    Y,
                    folds,
                    alpha_grid,
                    n_inner_folds,
                    groups=neural_groups
                )
                losses.append(orig_r2_mean - r2m)
            losses = np.asarray(losses)
            pct = float((losses < leace_loss).mean() * 100.0)
            controls[family] = {
                "kind": "random_null",
                "null_losses": losses,
                "null_loss_mean": float(losses.mean()),
                "null_loss_std": float(losses.std()),
                "null_q05": float(np.percentile(losses, 5)),
                "null_q95": float(np.percentile(losses, 95)),
                "leace_z": float((leace_loss - losses.mean()) / (losses.std() + 1e-12)),
                "leace_percentile": pct,
                "exceeds": bool(pct >= control_percentile)
            }
        else:  # pca controls are deterministic, so a single point
            ctrl = build_matched_control(
                family,
                Xp_tr,
                stats,
                seed=seed
            )
            _, r2m, _ = _score_condition(
                ctrl.transform(neural_reps),
                Y,
                folds,
                alpha_grid,
                n_inner_folds,
                groups=neural_groups
            )
            point = orig_r2_mean - r2m
            controls[family] = {
                "kind": "pca_point",
                "point_loss": float(point),
                "leace_exceeds": bool(leace_loss >= point),
                "exceeds": bool(leace_loss >= point),
                "achieved": ctrl.meta
            }

    exceeds_all = all(c["exceeds"] for c in controls.values())
    percentiles = [c["leace_percentile"] for c in controls.values() if "leace_percentile" in c]

    # 6. geometry of the erasure on the neural representations
    geometry = {
        "effrank_orig": effective_rank(neural_reps),
        "effrank_erased": effective_rank(En),
        "anisotropy_orig": anisotropy(neural_reps),
        "anisotropy_erased": anisotropy(En),
        "variance_removed": float(total_variance(neural_reps) - total_variance(En)),
        "perturbation_norm": float(np.linalg.norm(
            np.asarray(neural_reps, np.float64) - np.asarray(En, np.float64),
            axis=1
        ).mean())
    }

    return {
        "removed_rank": int(eraser.removed_rank_),
        "leace": {
            "M_hash": eraser.m_hash(),
            "mu": eraser.mu_,
            "eig_floor_rtol": eraser.eig_floor_rtol
        },
        "erasure_check": erasure_check,
        "selectivity": selectivity,
        "brain": {
            "original_r2": orig_r2,
            "leace_r2": leace_r2,
            "original_r2_mean": orig_r2_mean,
            "leace_r2_mean": leace_r2_mean,
            "original_r_mean": orig_r_mean,
            "leace_r_mean": leace_r_mean,
            "raw_loss_r2": raw_loss_r2,
            "raw_loss_r2_mean": float(leace_loss),
            "n_outer_folds": n_outer_folds,
            "n_inner_folds": n_inner_folds
        },
        "controls": controls,
        "control_relative": {
            "exceeds_all": bool(exceeds_all),
            "min_percentile": float(min(percentiles)) if percentiles else None,
            "threshold_percentile": float(control_percentile)
        },
        "geometry": geometry,
        "claim_status": (
            "confirmatory_checks_pending_multiplicity_and_hierarchical_uncertainty"
            if confirmatory
            else "exploratory_representation_intervention_only"
        ),
        "config": {
            "n_outer_folds": n_outer_folds,
            "n_inner_folds": n_inner_folds,
            "n_control_seeds": n_control_seeds,
            "confirmatory": bool(confirmatory),
            "probe_grouped_split": probe_groups is not None,
            "scoring": "complete_out_of_fold_predictions",
            "control_families": list(control_families),
            "control_percentile": control_percentile,
            "seed": seed
        }
    }


def representation_effect_gate(
    *,
    per_seed_results: list[dict],
    ood_results: list[dict] | None = None,
    replication_min_seeds: int = 3,
    min_effect_r2: float = 0.0
) -> dict:
    """Decide whether a *representation-level correlate* claim is licensed by the evidence.

    This gate tests whether the erasure effect is a reliable representation-level
    correlate: (a) replicates across seeds, (b) generalizes out of distribution,
    and (c) exceeds ALL matched controls in-distribution.

    A ``False`` verdict means: the observed brain-score drop may be within the
    noise range of matched controls, or may not replicate.  Do NOT claim causal
    necessity, causal mechanism, or that the brain "uses" the erased subspace.
    Report the result as an association: the linear subspace correlates with
    brain-predictive variance.

    ``causal_claim_gate`` is an alias kept for existing callers; the name
    overstates what LEACE erasure can show.
    """
    ood_results = ood_results or []
    losses = [r["brain"]["raw_loss_r2_mean"] for r in per_seed_results]
    positive = [loss > min_effect_r2 for loss in losses]

    replicates_across_seeds = sum(positive) >= replication_min_seeds
    sign_unanimous = len(positive) > 0 and all(positive)
    exceeds_all_controls = (
        len(per_seed_results) > 0
        and all(r["control_relative"]["exceeds_all"] for r in per_seed_results)
    )
    generalizes_ood = len(ood_results) > 0 and all(
        (r["brain"]["raw_loss_r2_mean"] > min_effect_r2)
        and r["control_relative"]["exceeds_all"]
        for r in ood_results
    )
    supported = bool(
        replicates_across_seeds and exceeds_all_controls and generalizes_ood
    )

    return {
        # a representation correlate, not causal necessity
        "representation_effect_supported": supported,
        "replicates_across_seeds": bool(replicates_across_seeds),
        "sign_unanimous": bool(sign_unanimous),
        "generalizes_ood": bool(generalizes_ood),
        "exceeds_all_controls": bool(exceeds_all_controls),
        "n_seeds": len(per_seed_results),
        "n_positive_seeds": int(sum(positive)),
        "per_seed_losses": [float(x) for x in losses],
        "ood_losses": [float(r["brain"]["raw_loss_r2_mean"]) for r in ood_results],
        "replication_min_seeds": replication_min_seeds,
        "interpretation": (
            "representation-level correlate: the erased linear subspace "
            "covaries with brain-predictive variance.  "
            "This is NOT evidence of causal necessity or that the brain "
            "computes the erased representation."
        )
    }


# alias for existing callers; prefer representation_effect_gate
causal_claim_gate = representation_effect_gate
