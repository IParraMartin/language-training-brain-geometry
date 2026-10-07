"""Cross-model representational similarity, and rank/variance-matched subspace controls.

The spectrum analysis (``manifold_pressure``) gives a *within*-model view of representational geometry: a
layer's covariance spectrum summarised by effective rank, anisotropy,
participation ratio and power-law decay (``manifold_pressure.spectral_profile``).
What it cannot answer is the comparative question — whether two corpora shape
the same manifold — because every one of those statistics is a scalar summary
that two quite different geometries can share.

This module adds the comparative half:

``linear_cka``
    Centred-kernel alignment between two representations of the *same* stimuli.
    Invariant to orthogonal transforms and isotropic scaling, so it compares the
    geometry rather than the coordinate system two independently trained
    networks happen to have landed in. That invariance is the whole reason it is
    usable here: models trained from different seeds have no shared basis.

``rdm`` / ``rsa``
    Representational dissimilarity matrices and the rank correlation between
    them. RSA is weaker than CKA (it sees only the distance ordering) but it is
    the form that compares a model against *brain* data, whose dimensions are
    voxels and carry no correspondence to model units at all.

``subspace_ablation_controls``
    The control a plain PC ablation lacks. Deleting the top-$m$
    principal directions and re-scoring conflates two explanations — that those
    directions carry the alignment, or merely that a lot of variance was
    removed. Separating them needs both a rank-matched and a variance-matched
    reference; see ``subspace_ablation_controls`` for why one alone cannot.

All functions take representations as ``(n_stimuli, n_units)`` and are pure:
no fitting state, no global RNG use, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

__all__ = [
    "column_center",
    "linear_cka",
    "rdm",
    "rsa",
    "rsa_from_rdms",
    "square_rdm",
    "grouped_permutation_null",
    "permutation_p_value",
    "SubspaceControl",
    "subspace_ablation_controls"
]


def column_center(X: NDArray) -> NDArray:
    """Subtract the column mean, the centring both CKA and PCA assume."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D (n_stimuli, n_units) array, got shape {X.shape}")
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: NDArray, Y: NDArray) -> float:
    """Linear centred-kernel alignment between two representations of one stimulus set.

    Computed in the *feature* form ``||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)``
    rather than by building two n-by-n Gram matrices. The two are algebraically
    identical for the linear kernel, but the feature form costs
    ``O(n * p * q)`` instead of ``O(n^2 * p)`` and never materialises an
    n-by-n matrix.

    Args:
        X: ``(n_stimuli, p)`` representation.
        Y: ``(n_stimuli, q)`` representation of the *same* stimuli, same order.

    Returns:
        float: CKA in ``[0, 1]``; 1.0 when the two agree up to an orthogonal
        transform and isotropic scaling. ``nan`` if either side has no variance,
        because CKA is undefined there rather than zero.

    Raises:
        ValueError: If the two arrays describe different numbers of stimuli.
    """
    Xc, Yc = column_center(X), column_center(Y)
    if Xc.shape[0] != Yc.shape[0]:
        raise ValueError(
            f"stimulus counts differ: X has {Xc.shape[0]}, Y has {Yc.shape[0]}"
        )
    cross = float(np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2)
    norm_x = float(np.linalg.norm(Xc.T @ Xc, ord="fro"))
    norm_y = float(np.linalg.norm(Yc.T @ Yc, ord="fro"))
    if norm_x == 0.0 or norm_y == 0.0:
        return float("nan")
    return cross / (norm_x * norm_y)


def rdm(X: NDArray, metric: str = "correlation") -> NDArray:
    """Condensed representational dissimilarity matrix over stimuli.

    Args:
        X: ``(n_stimuli, n_units)`` representation.
        metric: Any ``scipy.spatial.distance.pdist`` metric. ``"correlation"``
            is the RSA convention and is scale-invariant per stimulus.

    Returns:
        NDArray: Condensed vector of length ``n*(n-1)/2``.
    """
    return pdist(np.asarray(X, dtype=np.float64), metric=metric)


def rsa(X: NDArray, Y: NDArray, metric: str = "correlation") -> float:
    """Spearman correlation between two representations' dissimilarity structure.

    Unlike :func:`linear_cka` this needs no correspondence between the two
    feature spaces, which is what makes it the right tool against brain data:
    voxels and model units are not commensurable, but the distances they induce
    over a shared stimulus set are.

    .. note::
       **Only the Euclidean metric is exactly rotation-invariant.** Correlation
       distance normalises each stimulus by its mean and spread *across
       features*, so rotating the feature space perturbs it (empirically
       rho ~ 0.999 rather than 1.0 under a random rotation). That is harmless
       in the intended use — model against brain, where the brain's basis is
       fixed — but it makes correlation-RSA the wrong choice for comparing two
       independently trained networks, which share no basis. Use
       :func:`linear_cka` for model-to-model comparison, or pass
       ``metric="euclidean"`` if RSA is required there.

    Returns:
        float: Spearman rho over the condensed RDMs; ``nan`` if either RDM is
        constant.
    """
    return rsa_from_rdms(rdm(X, metric), rdm(Y, metric))


def rsa_from_rdms(a: NDArray, b: NDArray) -> float:
    """Spearman correlation between two already-computed condensed RDMs.

    The entry point for the common case where one side is fixed across many
    comparisons -- a brain RDM compared against every model and layer. Building
    it inside :func:`rsa` rebuilds that invariant object once per call; against
    27 models x 12 blocks x 7 voxel sets that is 2268 rebuilds of 7 matrices.

    Args:
        a: Condensed RDM, e.g. from :func:`rdm`.
        b: Condensed RDM over the same stimuli, same order and length.

    Returns:
        float: Spearman rho, or ``nan`` if either RDM is constant.

    Raises:
        ValueError: If the RDMs have different lengths, which means they do not
            describe the same stimulus set.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"RDMs describe different stimulus sets: {a.shape} vs {b.shape}")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


@dataclass(frozen=True)
class SubspaceControl:
    """One arm of a subspace-ablation comparison.

    Attributes:
        name: Arm label (``"top"``, ``"random"``, ``"bottom_variance_matched"``).
        n_directions_removed: Rank of the removed subspace.
        variance_removed_fraction: Share of total variance the arm deletes.
        variance_match_achieved: Whether this arm actually reached the top arm's
            variance. Always ``True`` for ``top``; meaningless for ``random``
            (recorded ``False``); and **not guaranteed** for
            ``bottom_variance_matched`` — see
            :func:`subspace_ablation_controls`. Check it before describing that
            arm as variance-matched in any result.
        projector: ``(p, p)`` matrix that removes the subspace, as ``X @ projector``.
    """

    name: str
    n_directions_removed: int
    variance_removed_fraction: float
    variance_match_achieved: bool
    projector: NDArray = field(repr=False)


def subspace_ablation_controls(
    X: NDArray,
    m: int,
    *,
    rng: np.random.Generator
) -> list[SubspaceControl]:
    """Build the top-$m$ ablation and the two references needed to read it.

    Removing the top $m$ principal directions and watching alignment fall admits
    two readings: those directions carried the alignment, or *any* comparable
    deletion would have hurt. Telling them apart needs two different controls,
    and neither suffices alone:

    - **rank-matched** (``random``): a uniformly random $m$-dimensional subspace.
      Same rank, but it removes far less variance, so on its own it cannot
      distinguish "these directions mattered" from "this much variance mattered".
    - **variance-matched** (``bottom_variance_matched``): the smallest set of
      *lowest*-variance principal directions whose total variance is at least
      that of the top $m$. Same variance, higher rank. Exact variance matching
      at equal rank is impossible in principle — the top $m$ directions carry the
      maximum variance any $m$-dimensional subspace can — which is precisely why
      the rank-matched and variance-matched arms must be reported as a pair.

    An alignment drop that exceeds *both* arms is evidence about those
    directions. A drop that either arm reproduces is evidence about magnitude.

    .. warning::
       The variance-matched arm **can be infeasible**. When the leading $m$
       directions carry more variance than every remaining direction combined —
       common for a sharply concentrated spectrum, which is exactly the regime
       this analysis targets — no set of trailing directions can reach it. The
       arm is then returned with ``variance_match_achieved=False`` and the most
       variance the trailing directions can supply. It is *not* clamped
       silently: an arm labelled variance-matched that removed 62% against the
       top arm's 88% would misreport the control as passed. Callers must
       propagate the flag and refrain from the variance-matched reading when it
       is ``False``, falling back to the rank-matched arm alone.

    Args:
        X: ``(n_stimuli, n_units)`` representation, centred internally.
        m: Number of leading principal directions to remove; ``1 <= m < n_units``.
        rng: Seeded generator; the random arm is the only stochastic part.

    Returns:
        list[SubspaceControl]: ``[top, random, bottom_variance_matched]``.

    Raises:
        ValueError: If ``m`` is outside ``[1, n_units)``.
    """
    Xc = column_center(X)
    n_units = Xc.shape[1]
    if not 1 <= m < n_units:
        raise ValueError(f"m must satisfy 1 <= m < n_units={n_units}, got {m}")

    # Right singular vectors are the principal directions; singular values
    # squared are proportional to the variance along each.
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = sv ** 2
    total = float(var.sum())
    if total == 0.0:
        raise ValueError("representation has zero variance; ablation is undefined")

    def remover(directions: NDArray) -> NDArray:
        """Projector removing the span of `directions` (rows, orthonormal)."""
        return np.eye(n_units) - directions.T @ directions

    top_dirs = Vt[:m]
    top_var = float(var[:m].sum()) / total

    # Rank-matched random subspace: orthonormalise a Gaussian draw. A Gaussian
    # matrix has full rank almost surely, so the QR factor is a uniformly random
    # orthonormal frame.
    gaussian = rng.standard_normal((n_units, m))
    q, _ = np.linalg.qr(gaussian)
    rand_dirs = q.T
    rand_var = float(np.sum((Xc @ rand_dirs.T) ** 2)) / total

    # Variance-matched: accumulate from the smallest eigenvalue upward until the
    # top-m variance is reached. Ascending order makes this the *smallest* such
    # set, so the arm removes as little rank as possible for the matched variance.
    #
    # The cap at n_units - 1 is what makes the match potentially unreachable: we
    # may never remove the whole space, or the ablation is not an ablation. When
    # the leading directions dominate hard enough, even every trailing direction
    # falls short, and that has to be reported rather than clamped away.
    ascending = var[::-1]
    cumulative = np.cumsum(ascending) / total
    k = int(np.searchsorted(cumulative, top_var, side="left") + 1)
    k = min(k, n_units - 1)
    bottom_dirs = Vt[::-1][:k]
    bottom_var = float(var[::-1][:k].sum()) / total
    matched = bool(bottom_var >= top_var - 1e-9)

    return [
        SubspaceControl(
            "top",
            m,
            top_var,
            True,
            remover(top_dirs)
        ),
        SubspaceControl(
            "random",
            m,
            rand_var,
            False,
            remover(rand_dirs)
        ),
        SubspaceControl(
            "bottom_variance_matched",
            k,
            bottom_var,
            matched,
            remover(bottom_dirs)
        )
    ]


def square_rdm(X: NDArray, metric: str = "correlation") -> NDArray:
    """Square form of :func:`rdm`, needed when stimuli must be re-indexed."""
    return squareform(rdm(X, metric))


def grouped_permutation_null(
    model_square: NDArray,
    target_square: NDArray,
    groups: list[str],
    *,
    rng: np.random.Generator,
    n_permutations: int = 1000
) -> NDArray:
    """Null distribution of RSA under stimulus relabelling *within* groups.

    The reference the encoding analyses lack. A raw RSA value has no scale: a
    correlation of 0.14 between two 243-stimulus dissimilarity structures is not
    obviously different from what arbitrary structures of the same shape would
    give. This resamples the stimulus-to-response correspondence to say so.

    **Why within groups.** Pereira's stimuli are nested in passages, and both
    model and brain geometry carry strong passage-block structure — stimuli from
    one passage resemble each other in both. An unrestricted permutation would
    destroy that block structure, so almost any real pairing would beat the null
    and the test would be trivially passed. Permuting only *within* passage
    preserves the block structure on both sides and asks the sharper question:
    given that both agree at the passage level, do they also agree about which
    stimulus is which? That is the conservative choice, and it is the grouping
    the benchmark itself cross-validates over (``split_coord="story"``).

    Args:
        model_square: ``(n, n)`` model RDM, e.g. from :func:`square_rdm`.
        target_square: ``(n, n)`` target RDM over the same stimuli, same order.
        groups: Length-``n`` group label per stimulus (passage or story).
        rng: Seeded generator.
        n_permutations: Number of draws.

    Returns:
        NDArray: ``(n_permutations,)`` null RSA values.

    Raises:
        ValueError: If the shapes or the group vector disagree, or if no group
            has more than one member (nothing could be permuted, so the "null"
            would be the observed value repeated).
    """
    model_square = np.asarray(model_square, dtype=np.float64)
    target_square = np.asarray(target_square, dtype=np.float64)
    n = model_square.shape[0]
    if model_square.shape != target_square.shape or model_square.shape[0] != model_square.shape[1]:
        raise ValueError(f"expected two equal square RDMs, got {model_square.shape} and {target_square.shape}")
    if len(groups) != n:
        raise ValueError(f"groups has length {len(groups)}, expected {n}")

    positions: dict[str, NDArray] = {}
    for index, label in enumerate(groups):
        positions.setdefault(label, []).append(index)
    blocks = [np.asarray(v) for v in positions.values()]
    if all(len(b) < 2 for b in blocks):
        raise ValueError(
            "every group has fewer than 2 stimuli; within-group permutation "
            "cannot move anything and the null would be degenerate"
        )

    target_condensed = squareform(target_square, checks=False)
    null = np.empty(n_permutations, dtype=np.float64)
    for draw in range(n_permutations):
        order = np.arange(n)
        for block in blocks:
            order[block] = rng.permutation(block)
        shuffled = model_square[np.ix_(order, order)]
        null[draw] = rsa_from_rdms(squareform(shuffled, checks=False), target_condensed)
    return null


def permutation_p_value(observed: float, null: NDArray) -> float:
    """One-sided p-value with the observed value included in the null count.

    The ``(count + 1) / (n + 1)`` form, not ``count / n``: the latter can report
    p = 0, which claims more resolution than the number of draws supports.
    """
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return float("nan")
    return float((np.sum(null >= observed) + 1) / (null.size + 1))
