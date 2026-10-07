"""Unit tests for cross-model representational similarity and subspace controls.

CPU-only, no downloads. The invariance tests are the load-bearing ones: CKA is
used here to compare independently trained networks that share no basis, so if
it were not invariant to rotation it would report seed differences as corpus
differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain_code.experiments.representational_similarity import (
    linear_cka,
    rdm,
    rsa,
    subspace_ablation_controls
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260819)


@pytest.fixture
def X(rng: np.random.Generator) -> np.ndarray:
    # Low-rank plus noise, so the spectrum is not flat and PC ablation has
    # something to remove.
    latent = rng.standard_normal((60, 4))
    mixing = rng.standard_normal((4, 24))
    return latent @ mixing + 0.1 * rng.standard_normal((60, 24))


class TestLinearCKA:
    def test_self_similarity_is_one(self, X):
        assert linear_cka(X, X) == pytest.approx(1.0)

    def test_symmetric(self, X, rng):
        Y = rng.standard_normal(X.shape)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X))

    def test_invariant_to_orthogonal_transform(self, X, rng):
        """The property the cross-model comparison depends on.

        Two networks trained from different seeds land in different coordinate
        systems. If CKA moved under rotation, every cross-model comparison would
        confound basis with geometry.
        """
        q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], X.shape[1])))
        assert linear_cka(X, X @ q) == pytest.approx(1.0)

    def test_invariant_to_isotropic_scaling(self, X):
        assert linear_cka(X, 7.3 * X) == pytest.approx(1.0)

    def test_invariant_to_translation(self, X):
        assert linear_cka(X, X + 12.5) == pytest.approx(1.0)

    def test_independent_representations_score_low(self, rng):
        a = rng.standard_normal((200, 30))
        b = rng.standard_normal((200, 30))
        assert linear_cka(a, b) < 0.35

    def test_bounded_in_unit_interval(self, X, rng):
        for _ in range(10):
            Y = rng.standard_normal((X.shape[0], rng.integers(3, 40)))
            value = linear_cka(X, Y)
            assert 0.0 <= value <= 1.0 + 1e-12

    def test_differing_feature_counts_allowed(self, X, rng):
        assert np.isfinite(linear_cka(X, rng.standard_normal((X.shape[0], 5))))

    def test_mismatched_stimulus_count_raises(self, X, rng):
        with pytest.raises(ValueError, match="stimulus counts differ"):
            linear_cka(X, rng.standard_normal((X.shape[0] + 1, 8)))

    def test_zero_variance_is_nan_not_zero(self, X):
        assert np.isnan(linear_cka(X, np.ones_like(X)))

    def test_rejects_non_2d(self, rng):
        with pytest.raises(ValueError, match="2-D"):
            linear_cka(rng.standard_normal(10), rng.standard_normal(10))


class TestRSA:
    def test_self_similarity_is_one(self, X):
        assert rsa(X, X) == pytest.approx(1.0)

    def test_euclidean_is_exactly_rotation_invariant(self, X, rng):
        q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], X.shape[1])))
        assert rsa(X, X @ q, metric="euclidean") == pytest.approx(1.0, abs=1e-9)

    def test_correlation_is_only_approximately_rotation_invariant(self, X, rng):
        """Documents a real limitation, not a tolerance fudge.

        Correlation distance normalises each stimulus across features, so a
        rotation of the feature space perturbs it. This is why cross-model
        comparison uses CKA rather than correlation-RSA: two seeds share no
        basis, and correlation-RSA would charge them for that.
        """
        q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], X.shape[1])))
        value = rsa(X, X @ q, metric="correlation")
        assert value < 1.0 - 1e-6
        assert value > 0.99

    def test_works_across_incommensurable_feature_spaces(self, X, rng):
        """The brain case: 24 model units against 100 'voxels'."""
        voxels = X @ rng.standard_normal((X.shape[1], 100))
        assert rsa(X, voxels) > 0.9

    def test_independent_representations_score_near_zero(self, rng):
        a, b = rng.standard_normal((80, 20)), rng.standard_normal((80, 20))
        assert abs(rsa(a, b)) < 0.25

    def test_condensed_rdm_length(self, X):
        n = X.shape[0]
        assert rdm(X).shape == (n * (n - 1) // 2,)

    def test_constant_representation_is_nan(self, X):
        assert np.isnan(rsa(X, np.zeros_like(X)))


class TestSubspaceAblationControls:
    def test_returns_three_named_arms(self, X, rng):
        names = [c.name for c in subspace_ablation_controls(X, 3, rng=rng)]
        assert names == ["top", "random", "bottom_variance_matched"]

    def test_top_arm_removes_requested_rank(self, X, rng):
        top = subspace_ablation_controls(X, 3, rng=rng)[0]
        assert top.n_directions_removed == 3
        # A projector that removes a 3-D subspace of a 24-D space has rank 21.
        assert np.linalg.matrix_rank(top.projector) == X.shape[1] - 3

    def test_projectors_are_idempotent(self, X, rng):
        for control in subspace_ablation_controls(X, 3, rng=rng):
            P = control.projector
            assert np.allclose(P @ P, P, atol=1e-8), control.name

    def test_top_arm_removes_the_most_variance_at_equal_rank(self, X, rng):
        top, random_arm, _ = subspace_ablation_controls(X, 3, rng=rng)
        assert top.n_directions_removed == random_arm.n_directions_removed
        assert top.variance_removed_fraction > random_arm.variance_removed_fraction

    def test_variance_match_costs_rank_when_it_is_feasible(self, rng):
        """The documented trade: equal variance is only buyable with more rank.

        Needs a spectrum flat enough for the trailing directions to reach the
        leading ones, so this fixture is deliberately near-isotropic.
        """
        flat = rng.standard_normal((200, 24))
        top, _, matched = subspace_ablation_controls(flat, 3, rng=rng)
        assert matched.variance_match_achieved
        assert matched.variance_removed_fraction >= top.variance_removed_fraction - 1e-9
        assert matched.n_directions_removed > top.n_directions_removed

    def test_infeasible_variance_match_is_flagged_not_clamped(self, X, rng):
        """A concentrated spectrum makes the variance-matched arm unreachable.

        In `X` the top 3 directions carry ~88% of the variance while all 21
        remaining carry ~12%, so no set of trailing directions can match. The
        arm must come back flagged: an arm silently labelled variance-matched
        that removed far less variance would report a control as passed when it
        never ran.
        """
        top, _, matched = subspace_ablation_controls(X, 3, rng=rng)
        assert top.variance_removed_fraction > 0.8
        assert not matched.variance_match_achieved
        assert matched.variance_removed_fraction < top.variance_removed_fraction

    def test_top_arm_is_always_trivially_matched(self, X, rng):
        assert subspace_ablation_controls(X, 3, rng=rng)[0].variance_match_achieved

    def test_random_arm_never_claims_a_variance_match(self, X, rng):
        assert not subspace_ablation_controls(X, 3, rng=rng)[1].variance_match_achieved

    def test_ablation_actually_reduces_variance(self, X, rng):
        Xc = X - X.mean(axis=0)
        before = float((Xc ** 2).sum())
        for control in subspace_ablation_controls(X, 3, rng=rng):
            after = float(((Xc @ control.projector) ** 2).sum())
            expected = before * (1.0 - control.variance_removed_fraction)
            assert after == pytest.approx(expected, rel=1e-6), control.name

    def test_random_arm_varies_with_seed(self, X):
        a = subspace_ablation_controls(X, 3, rng=np.random.default_rng(1))[1]
        b = subspace_ablation_controls(X, 3, rng=np.random.default_rng(2))[1]
        assert not np.allclose(a.projector, b.projector)

    def test_deterministic_given_a_seed(self, X):
        a = subspace_ablation_controls(X, 3, rng=np.random.default_rng(7))[1]
        b = subspace_ablation_controls(X, 3, rng=np.random.default_rng(7))[1]
        assert np.allclose(a.projector, b.projector)

    @pytest.mark.parametrize("bad_m", [0, -1, 24, 99])
    def test_out_of_range_m_raises(
        self,
        X,
        rng,
        bad_m
    ):
        with pytest.raises(ValueError, match="m must satisfy"):
            subspace_ablation_controls(X, bad_m, rng=rng)

    def test_zero_variance_input_raises(self, rng):
        with pytest.raises(ValueError, match="zero variance"):
            subspace_ablation_controls(np.ones((10, 5)), 2, rng=rng)


class TestRSAFromRDMs:
    """The hoisted entry point must agree exactly with the all-in-one form."""

    def test_agrees_with_rsa(self, X, rng):
        from brain_code.experiments.representational_similarity import rdm, rsa_from_rdms

        Y = X @ rng.standard_normal((X.shape[1], 40))
        assert rsa_from_rdms(rdm(X), rdm(Y)) == pytest.approx(rsa(X, Y))

    def test_self_similarity_is_one(self, X):
        from brain_code.experiments.representational_similarity import rdm, rsa_from_rdms

        assert rsa_from_rdms(rdm(X), rdm(X)) == pytest.approx(1.0)

    def test_mismatched_rdm_lengths_raise(self, X, rng):
        from brain_code.experiments.representational_similarity import rdm, rsa_from_rdms

        with pytest.raises(ValueError, match="different stimulus sets"):
            rsa_from_rdms(rdm(X), rdm(rng.standard_normal((X.shape[0] + 3, 9))))

    def test_constant_rdm_is_nan(self, X):
        from brain_code.experiments.representational_similarity import rdm, rsa_from_rdms

        assert np.isnan(rsa_from_rdms(rdm(X), rdm(np.zeros_like(X))))


class TestGroupedPermutationNull:
    """The reference that turns a raw RSA number into a testable claim."""

    @staticmethod
    def _setup(
        rng,
        n_groups=8,
        per_group=6,
        noise=0.3
    ):
        from brain_code.experiments.representational_similarity import square_rdm

        groups, rows = [], []
        for g in range(n_groups):
            centre = rng.standard_normal(12) * 3.0  # strong block structure
            for _ in range(per_group):
                rows.append(centre + rng.standard_normal(12) * 0.5)
                groups.append(f"g{g}")
        X = np.asarray(rows)
        Y = X @ rng.standard_normal((12, 30)) + noise * rng.standard_normal((len(X), 30))
        return square_rdm(X), square_rdm(Y), groups

    def test_null_has_requested_size(self, rng):
        from brain_code.experiments.representational_similarity import grouped_permutation_null

        a, b, groups = self._setup(rng)
        assert grouped_permutation_null(
            a,
            b,
            groups,
            rng=rng,
            n_permutations=50
        ).shape == (50,)

    def test_true_correspondence_beats_the_null(self, rng):
        from brain_code.experiments.representational_similarity import (
            grouped_permutation_null,
            permutation_p_value,
            rsa_from_rdms
        )
        from scipy.spatial.distance import squareform

        a, b, groups = self._setup(rng)
        observed = rsa_from_rdms(squareform(a, checks=False), squareform(b, checks=False))
        null = grouped_permutation_null(
            a,
            b,
            groups,
            rng=rng,
            n_permutations=200
        )
        assert observed > null.max()
        assert permutation_p_value(observed, null) < 0.01

    def test_null_retains_block_structure_so_it_is_not_near_zero(self, rng):
        """The conservatism check.

        Within-group permutation preserves passage-level agreement, so the null
        should sit well above zero. A null centred on zero would mean the block
        structure had been destroyed and the test had become trivial.
        """
        from brain_code.experiments.representational_similarity import grouped_permutation_null

        a, b, groups = self._setup(rng)
        null = grouped_permutation_null(
            a,
            b,
            groups,
            rng=rng,
            n_permutations=200
        )
        assert null.mean() > 0.2

    def test_unrelated_structures_do_not_beat_the_null(self, rng):
        from brain_code.experiments.representational_similarity import (
            grouped_permutation_null,
            permutation_p_value,
            rsa_from_rdms,
            square_rdm
        )
        from scipy.spatial.distance import squareform

        a, _, groups = self._setup(rng)
        unrelated = square_rdm(rng.standard_normal((len(groups), 30)))
        observed = rsa_from_rdms(squareform(a, checks=False), squareform(unrelated, checks=False))
        null = grouped_permutation_null(
            a,
            unrelated,
            groups,
            rng=rng,
            n_permutations=200
        )
        assert permutation_p_value(observed, null) > 0.05

    def test_deterministic_given_a_seed(self, rng):
        from brain_code.experiments.representational_similarity import grouped_permutation_null

        a, b, groups = self._setup(rng)
        one = grouped_permutation_null(
            a,
            b,
            groups,
            rng=np.random.default_rng(3),
            n_permutations=25
        )
        two = grouped_permutation_null(
            a,
            b,
            groups,
            rng=np.random.default_rng(3),
            n_permutations=25
        )
        np.testing.assert_allclose(one, two)

    def test_singleton_groups_raise(self, rng):
        from brain_code.experiments.representational_similarity import grouped_permutation_null

        a, b, _ = self._setup(rng, n_groups=6, per_group=1)
        with pytest.raises(ValueError, match="degenerate"):
            grouped_permutation_null(
                a,
                b,
                [f"g{i}" for i in range(6)],
                rng=rng,
                n_permutations=10
            )

    def test_mismatched_group_length_raises(self, rng):
        from brain_code.experiments.representational_similarity import grouped_permutation_null

        a, b, groups = self._setup(rng)
        with pytest.raises(ValueError, match="groups has length"):
            grouped_permutation_null(
                a,
                b,
                groups[:-1],
                rng=rng,
                n_permutations=10
            )

    def test_p_value_never_reports_zero(self):
        from brain_code.experiments.representational_similarity import permutation_p_value

        assert permutation_p_value(99.0, np.zeros(100)) == pytest.approx(1 / 101)

    def test_p_value_ignores_nan_draws(self):
        from brain_code.experiments.representational_similarity import permutation_p_value

        null = np.array([
            0.1,
            np.nan,
            0.2,
            0.3
        ])
        assert permutation_p_value(0.25, null) == pytest.approx(2 / 4)
