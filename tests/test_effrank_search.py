"""Tests for the spectrum-slice shortcut in the `pca_effrank` matched control.

The control searches for the number of principal directions whose removal
leaves an effective rank closest to LEACE's. Removing the top m directions
zeroes exactly the top m singular values, so the sweep reads slices of one
spectrum instead of running one eigendecomposition per candidate. These tests
pin that identity and the cap at the matrix rank.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain_code.experiments.geometry import effective_rank, effective_rank_from_spectrum, spectrum
from brain_code.experiments.subspace_erasure import _pca_removal


def _centred(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return X - X.mean(axis=0)


@pytest.mark.parametrize("m", [1, 2, 5, 50, 100, 400])
def test_slice_matches_explicit_projection(m):
    """Slicing the spectrum equals projecting then re-decomposing."""
    Xc = _centred(450, 768)
    cov = Xc.T @ Xc / len(Xc)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]

    explicit = effective_rank(Xc @ _pca_removal(evecs, m))
    sliced = effective_rank_from_spectrum(spectrum(Xc)[m:])
    # float64 rounding through an eigendecomposition of a rank-deficient
    # product; the sliced value is if anything the better-conditioned one
    assert sliced == pytest.approx(explicit, rel=1e-5)


def test_search_stays_within_the_matrix_rank():
    """Beyond the rank there is no spectrum left to slice.

    A (450, 768) matrix has at most 450 singular values, so removing 700
    directions leaves nothing but rounding error. An explicit projection would
    still score that region, reporting an effective rank of several hundred for
    a matrix that cannot exceed 68, and let a meaningless m win the search.
    """
    Xc = _centred(450, 768)
    sv = spectrum(Xc)
    assert len(sv) == 450
    assert np.isnan(effective_rank_from_spectrum(sv[700:]))


def test_effective_rank_from_spectrum_matches_effective_rank():
    rng = np.random.default_rng(3)
    for shape in [(60, 120), (120, 60), (80, 80)]:
        X = rng.standard_normal(shape)
        assert effective_rank_from_spectrum(spectrum(X)) == pytest.approx(
            effective_rank(X), rel=1e-12
        )


def test_effective_rank_from_spectrum_edge_cases():
    assert np.isnan(effective_rank_from_spectrum(np.array([])))
    assert np.isnan(effective_rank_from_spectrum(np.zeros(5)))
    # a flat spectrum of k values is maximally isotropic: effective rank k
    assert effective_rank_from_spectrum(np.ones(7)) == pytest.approx(7.0)
    # a single direction is rank-1 regardless of scale
    assert effective_rank_from_spectrum(np.array([4.2])) == pytest.approx(1.0)


def test_entropy_is_order_invariant():
    """Slices arrive descending; the measure must not depend on that."""
    rng = np.random.default_rng(5)
    sv = np.abs(rng.standard_normal(40))
    assert effective_rank_from_spectrum(sv) == pytest.approx(
        effective_rank_from_spectrum(sv[::-1])
    )
