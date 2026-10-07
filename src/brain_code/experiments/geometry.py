"""Self-contained spectral geometry metrics for representation matrices.

These helpers mirror the embedding-geometry metrics in
``src/brain_code/training/checkpoint_logger.py`` (``embedding_anisotropy`` /
``embedding_effective_rank``), but are deliberately re-implemented here so the
analysis modules (LEACE erasure and its matched controls) do NOT import from the
training package — ``checkpoint_logger`` imports ``transformers`` at module
top, and pulling the whole transformers stack into a pure-numpy analysis path
would slow every unit test.

All functions operate on a 2-D matrix ``X`` of shape ``(n_samples, n_features)``
and use float64 SVD for numerical quality.  ``effective_rank`` and
``anisotropy`` are scale-invariant (they normalise the singular-value
spectrum), so they are directly comparable across conditions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spectrum(X: NDArray) -> NDArray:
    """Return the singular values of ``X`` in descending order (float64).

    Computed as ``sqrt(eigvalsh(G))`` where ``G`` is the smaller of the two Gram
    matrices (``X Xᵀ`` when ``n <= p``, else ``Xᵀ X``), which is mathematically
    identical to the singular values but 2-3x faster than a full SVD at the
    shapes used here; the matched-control search calls this hundreds of times
    per run.

    Deliberately **not** offloaded to CUDA. SVD and symmetric eigendecomposition
    are iterative and parallelize poorly, and float64 cuSOLVER on a consumer card
    measured ~10x *slower* than NumPy at these shapes.

    Args:
        X: Matrix of shape ``(n_samples, n_features)``.

    Returns:
        NDArray: ``min(n_samples, n_features)`` singular values, descending.
    """
    Xm = np.asarray(X, dtype=np.float64)
    n, p = Xm.shape
    gram = Xm @ Xm.T if n <= p else Xm.T @ Xm
    evals = np.linalg.eigvalsh(gram)[::-1]
    # rounding can push near-zero eigenvalues slightly negative
    return np.sqrt(np.clip(evals, 0.0, None))


def anisotropy(X: NDArray) -> float:
    """Fraction of spectral mass in the leading singular direction.

    0 = perfectly isotropic; 1 = rank-1.  Returns NaN for an all-zero matrix.
    """
    sv = spectrum(X)
    total = sv.sum()
    if total <= 0:
        return float("nan")
    return float(sv[0] / total)


def effective_rank_from_spectrum(sv: NDArray) -> float:
    """Effective rank of an already-computed singular-value spectrum.

    Split out so callers that sweep a family of related matrices can compute the
    spectrum once instead of once per matrix. Removing the top ``m`` principal
    directions, for instance, zeroes exactly the top ``m`` singular values and
    leaves the rest untouched, so the whole sweep reads slices of one spectrum.

    Args:
        sv: Singular values, any order (entropy is order-invariant).

    Returns:
        float: ``exp(Shannon entropy of the normalised spectrum)``, or NaN if
        the spectrum sums to zero.
    """
    sv = np.asarray(sv, dtype=np.float64)
    total = sv.sum()
    if total <= 0:
        return float("nan")
    probs = sv / total
    h = float(-np.sum(probs * np.log(probs + 1e-15)))
    return float(np.exp(h))


def effective_rank(X: NDArray) -> float:
    """``exp(Shannon entropy of the normalised singular-value spectrum)``.

    Ranges from 1 (rank-1) to ``min(n_samples, n_features)`` (isotropic).
    This is the Roy & Vetterli (2007) effective rank.  Returns NaN for an
    all-zero matrix.
    """
    return effective_rank_from_spectrum(spectrum(X))


def total_variance(X: NDArray) -> float:
    """Total variance = trace of the (column) covariance of ``X``.

    Equivalent to ``sum(var(X, axis=0))``; measures how much activation
    variance a projection removes.
    """
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n = max(len(Xc), 1)
    return float((Xc * Xc).sum() / n)
