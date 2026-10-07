"""Tests for LEACE concept erasure + matched controls + orchestrator."""

from __future__ import annotations


import numpy as np
import pytest

from brain_code.experiments.probes import (
    chance_level,
    linear_probe_accuracy,
    nonlinear_probe_accuracy
)
from brain_code.experiments.subspace_erasure import (
    CONTROL_FAMILIES,
    LeaceEraser,
    build_matched_control,
    causal_claim_gate,
    cross_fitted_erasure_analysis,
    leace_realized_stats
)


def _concept_data(
    n=400,
    d=12,
    k=3,
    sep=2.0,
    signal_dims=4,
    seed=0
):
    """k-class concept living in the first ``signal_dims`` dimensions."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    means = rng.standard_normal((k, d)) * sep
    means[:, signal_dims:] = 0.0
    X = means[y] + rng.standard_normal((n, d))
    return X.astype(np.float64), y


def _onehot(y):
    classes = np.unique(y)
    return np.eye(len(classes))[np.searchsorted(classes, y)]


def test_leace_zeroes_linear_covariance():
    X, y = _concept_data(seed=1)
    er = LeaceEraser().fit(X, y)
    E = er.transform(X)
    Zc = _onehot(y) - _onehot(y).mean(0)
    Ec = E - E.mean(0)
    cov = Ec.T @ Zc / len(X)
    assert np.abs(cov).max() < 1e-8


def test_leace_zeroes_linear_probe_accuracy():
    X, y = _concept_data(k=2, seed=2)
    er = LeaceEraser().fit(X, y)
    E = er.transform(X)
    # Fresh linear probe on erased reps should be ~chance.
    acc = linear_probe_accuracy(
        E[:300],
        y[:300],
        E[300:],
        y[300:]
    )
    ch = chance_level(y[300:])
    assert acc <= ch["majority"] + 0.1


@pytest.mark.parametrize("k", [2, 3, 5])
def test_leace_removed_rank_is_k_minus_1(k):
    X, y = _concept_data(k=k, signal_dims=min(k + 2, 8), seed=k)
    er = LeaceEraser().fit(X, y)
    assert er.removed_rank_ == k - 1


def test_leace_preserves_orthogonal_concept():
    """A concept in a subspace orthogonal to the target stays decodable."""
    rng = np.random.default_rng(3)
    n, d = 400, 12
    y_target = rng.integers(0, 2, n)
    y_other = rng.integers(0, 2, n)
    X = np.zeros((n, d))
    X[:, 0] = 3.0 * y_target + rng.standard_normal(n)      # target in dim 0
    X[:, 5] = 3.0 * y_other + rng.standard_normal(n)       # other in dim 5
    X[:, 1:5] = rng.standard_normal((n, 4))
    X[:, 6:] = rng.standard_normal((n, d - 6))
    er = LeaceEraser().fit(X, y_target)
    E = er.transform(X)
    acc_other = linear_probe_accuracy(
        E[:300],
        y_other[:300],
        E[300:],
        y_other[300:]
    )
    assert acc_other > 0.85       # unrelated concept preserved


def test_leace_idempotent_on_centered_data():
    X, y = _concept_data(seed=4)
    er = LeaceEraser().fit(X, y)
    M = er.M_
    assert np.linalg.norm(M @ M - M) < 1e-8


def test_leace_transpose_guard():
    """Using M (not M.T) must NOT zero the covariance — regression guard."""
    X, y = _concept_data(k=2, seed=5)
    er = LeaceEraser().fit(X, y)
    Zc = _onehot(y) - _onehot(y).mean(0)
    right = (X - er.mu_) @ er.M_.T + er.mu_
    wrong = (X - er.mu_) @ er.M_ + er.mu_
    cov_right = np.abs((right - right.mean(0)).T @ Zc / len(X)).max()
    cov_wrong = np.abs((wrong - wrong.mean(0)).T @ Zc / len(X)).max()
    assert cov_right < 1e-8
    assert cov_wrong > 1e-3


def test_leace_affine_preserves_global_mean():
    X, y = _concept_data(seed=6)
    er = LeaceEraser().fit(X, y)
    E = er.transform(X)
    np.testing.assert_allclose(E.mean(0), X.mean(0), atol=1e-8)


def test_leace_rank_deficient_covariance():
    """Collinear features must not break the whitening; guarantee still holds."""
    X, y = _concept_data(seed=7)
    X[:, 3] = X[:, 2]          # exact collinearity
    er = LeaceEraser().fit(X, y)
    E = er.transform(X)
    Zc = _onehot(y) - _onehot(y).mean(0)
    cov = np.abs((E - E.mean(0)).T @ Zc / len(X)).max()
    assert cov < 1e-7


def test_leace_float32_in_float32_out():
    X, y = _concept_data(k=2, seed=8)
    er = LeaceEraser().fit(X, y)
    E = er.transform(X.astype(np.float32))
    assert E.dtype == np.float32
    Zc = _onehot(y) - _onehot(y).mean(0)
    cov = np.abs((E - E.mean(0)).T @ Zc / len(X)).max()
    assert cov < 1e-5          # float32 precision floor


def test_leace_single_class_is_identity():
    X, _ = _concept_data(seed=9)
    y = np.zeros(len(X), dtype=int)      # one class → nothing to erase
    with pytest.warns(UserWarning):
        er = LeaceEraser().fit(X, y)
    assert er.removed_rank_ == 0
    np.testing.assert_allclose(er.transform(X), X, atol=1e-10)


def test_leace_matches_reference_concept_erasure():
    """Cross-check against EleutherAI concept-erasure (shrinkage disabled)."""
    ce = pytest.importorskip("concept_erasure")
    import torch

    X, y = _concept_data(
        n=300,
        d=8,
        k=3,
        sep=1.5,
        signal_dims=8,
        seed=10
    )
    ours = LeaceEraser().fit(X, y).transform(X)
    Xt = torch.tensor(X, dtype=torch.float64)
    zt = torch.tensor(_onehot(y), dtype=torch.float64)
    ref = ce.LeaceEraser.fit(Xt, zt, shrinkage=False)(Xt).numpy()
    assert np.abs(ours - ref).max() < 1e-6


def _stats():
    X, y = _concept_data(seed=11)
    er = LeaceEraser().fit(X, y)
    return X, er, leace_realized_stats(er, X)


@pytest.mark.parametrize("family", ["random_rank", "pca_rank"])
def test_control_rank_match(family):
    X, er, stats = _stats()
    d = X.shape[1]
    ctrl = build_matched_control(
        family,
        X,
        stats,
        seed=1
    )
    assert np.linalg.matrix_rank(ctrl.A) == d - stats["removed_rank"]


@pytest.mark.parametrize("family", ["random_var", "pca_var"])
def test_control_variance_match(family):
    X, er, stats = _stats()
    ctrl = build_matched_control(
        family,
        X,
        stats,
        seed=1
    )
    np.testing.assert_allclose(
        ctrl.meta["achieved_variance_removed"], stats["variance_removed"], rtol=0.02
    )


@pytest.mark.parametrize("family", ["random_norm", "pca_norm"])
def test_control_norm_match(family):
    X, er, stats = _stats()
    ctrl = build_matched_control(
        family,
        X,
        stats,
        seed=1
    )
    np.testing.assert_allclose(
        ctrl.meta["achieved_perturbation_norm"], stats["perturbation_norm"], rtol=1e-3
    )


def test_control_effrank_beats_rank_control_at_matching_effrank():
    X, er, stats = _stats()
    target = stats["effrank_after"]
    effrank_ctrl = build_matched_control(
        "pca_effrank",
        X,
        stats,
        seed=1
    )
    rank_ctrl = build_matched_control(
        "pca_rank",
        X,
        stats,
        seed=1
    )
    gap_eff = abs(effrank_ctrl.meta["achieved_effrank_after"] - target)
    gap_rank = abs(rank_ctrl.meta["achieved_effrank_after"] - target)
    assert gap_eff <= gap_rank + 1e-9


def test_control_frozen_applies_to_new_data():
    X, er, stats = _stats()
    ctrl = build_matched_control(
        "pca_rank",
        X,
        stats,
        seed=1
    )
    Xnew = np.random.default_rng(99).standard_normal((25, X.shape[1]))
    out = ctrl.transform(Xnew)
    assert out.shape == Xnew.shape


def test_all_control_families_build():
    X, er, stats = _stats()
    for family in CONTROL_FAMILIES:
        ctrl = build_matched_control(
            family,
            X,
            stats,
            seed=2
        )
        assert ctrl.A.shape == (X.shape[1], X.shape[1])


def test_linear_probe_recovers_signal_and_chance():
    X, y = _concept_data(k=2, signal_dims=4, seed=12)
    acc = linear_probe_accuracy(
        X[:300],
        y[:300],
        X[300:],
        y[300:]
    )
    assert acc > 0.9
    y_rand = np.random.default_rng(0).integers(0, 2, len(X))
    acc_rand = linear_probe_accuracy(
        X[:300],
        y_rand[:300],
        X[300:],
        y_rand[300:]
    )
    assert acc_rand < 0.7


def test_linear_probe_deterministic():
    X, y = _concept_data(k=2, seed=13)
    a1 = linear_probe_accuracy(
        X[:300],
        y[:300],
        X[300:],
        y[300:]
    )
    a2 = linear_probe_accuracy(
        X[:300],
        y[:300],
        X[300:],
        y[300:]
    )
    assert a1 == a2


def test_nonlinear_probe_learns_xor_linear_cannot():
    rng = np.random.default_rng(14)
    n, d = 600, 6
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y = ((x0 > 0) ^ (x1 > 0)).astype(int)     # XOR — not linearly separable
    X = np.column_stack([x0, x1, rng.standard_normal((n, d - 2))])
    tr, te = slice(0, 450), slice(450, n)
    lin = linear_probe_accuracy(
        X[tr],
        y[tr],
        X[te],
        y[te]
    )
    nonlin = nonlinear_probe_accuracy(
        X[tr],
        y[tr],
        X[te],
        y[te],
        epochs=300,
        seed=0
    )
    assert lin < 0.7
    assert nonlin > 0.8


def test_chance_level_majority():
    y = np.array([0] * 90 + [1] * 10)
    ch = chance_level(y)
    assert abs(ch["uniform"] - 0.5) < 1e-9
    assert abs(ch["majority"] - 0.9) < 1e-9


def _orchestrator_inputs(seed=15):
    rng = np.random.default_rng(seed)
    d, k = 12, 2
    n_probe = 200
    yp = np.tile([0, 1], n_probe // 2)
    pm = np.zeros((k, d))
    pm[1, 0], pm[1, 1] = 3.0, 2.0
    Xp = pm[yp] + rng.standard_normal((n_probe, d))
    unrel = rng.integers(0, 3, n_probe)
    perm = rng.permutation(n_probe)
    tr, te = perm[:150], perm[150:]
    n_stim, n_reg = 100, 5
    ys = rng.integers(0, k, n_stim)
    Xs = pm[ys] + rng.standard_normal((n_stim, d))
    W = rng.standard_normal((d, n_reg))
    Y = Xs @ W + rng.standard_normal((n_stim, n_reg)) * 0.5
    groups = np.arange(n_stim) // 4
    return dict(
        probe_reps=Xp,
        probe_labels=yp,
        probe_train_idx=tr,
        probe_test_idx=te,
        neural_reps=Xs,
        neural_responses=Y,
        neural_groups=groups,
        selectivity_labels={"unrelated": unrel},
        n_outer_folds=4,
        n_inner_folds=3,
        n_control_seeds=4,
        seed=seed
    )


def test_orchestrator_schema():
    res = cross_fitted_erasure_analysis(**_orchestrator_inputs())
    for key in [
        "removed_rank",
        "leace",
        "erasure_check",
        "selectivity",
        "brain",
        "controls",
        "control_relative",
        "geometry",
        "config"
    ]:
        assert key in res
    assert set(res["controls"]) == set(CONTROL_FAMILIES)
    assert "raw_loss_r2_mean" in res["brain"]


def test_orchestrator_null_shape():
    res = cross_fitted_erasure_analysis(**_orchestrator_inputs())
    for fam, c in res["controls"].items():
        if c["kind"] == "random_null":
            assert c["null_losses"].shape == (4,)


def test_orchestrator_reproducible():
    r1 = cross_fitted_erasure_analysis(**_orchestrator_inputs(seed=16))
    r2 = cross_fitted_erasure_analysis(**_orchestrator_inputs(seed=16))
    assert r1["leace"]["M_hash"] == r2["leace"]["M_hash"]
    assert r1["brain"]["raw_loss_r2_mean"] == r2["brain"]["raw_loss_r2_mean"]


def test_orchestrator_grouped_folds_no_leakage():
    from brain_code.experiments.subspace_erasure import _folds_from_groups
    groups = np.arange(40) // 4       # 10 groups of 4
    folds = _folds_from_groups(groups, n_folds=4, seed=0)
    for train_idx, test_idx in folds:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_orchestrator_detects_real_effect():
    """On data where the concept drives responses, LEACE loss beats controls."""
    res = cross_fitted_erasure_analysis(**_orchestrator_inputs(seed=18))
    assert res["brain"]["raw_loss_r2_mean"] > 0
    assert res["control_relative"]["exceeds_all"]


def _fake_result(loss, exceeds_all):
    return {"brain": {"raw_loss_r2_mean": loss}, "control_relative": {"exceeds_all": exceeds_all}}


def test_claim_gate_all_true():
    seeds = [_fake_result(0.2, True) for _ in range(3)]
    gate = causal_claim_gate(per_seed_results=seeds, ood_results=[_fake_result(0.15, True)])
    assert gate["representation_effect_supported"] is True


def test_claim_gate_fails_if_controls_not_exceeded():
    seeds = [_fake_result(0.2, True), _fake_result(0.2, False), _fake_result(0.2, True)]
    gate = causal_claim_gate(per_seed_results=seeds, ood_results=[_fake_result(0.15, True)])
    assert gate["exceeds_all_controls"] is False
    assert gate["representation_effect_supported"] is False


def test_claim_gate_fails_ood():
    seeds = [_fake_result(0.2, True) for _ in range(3)]
    gate = causal_claim_gate(per_seed_results=seeds, ood_results=[_fake_result(-0.01, True)])
    assert gate["generalizes_ood"] is False
    assert gate["representation_effect_supported"] is False


def test_claim_gate_requires_min_seeds():
    seeds = [_fake_result(0.2, True), _fake_result(0.2, True)]      # only 2
    gate = causal_claim_gate(
        per_seed_results=seeds,
        ood_results=[_fake_result(0.1, True)],
        replication_min_seeds=3
    )
    assert gate["replicates_across_seeds"] is False


def test_gate_returns_interpretation_field():
    """representation_effect_gate must include a non-empty interpretation field."""
    from brain_code.experiments.subspace_erasure import representation_effect_gate
    seeds = [{"brain": {"raw_loss_r2_mean": 0.1}, "control_relative": {"exceeds_all": True}} for _ in range(3)]
    gate = representation_effect_gate(
        per_seed_results=seeds,
        ood_results=[{"brain": {"raw_loss_r2_mean": 0.1}, "control_relative": {"exceeds_all": True}}]
    )
    assert "interpretation" in gate
    assert len(gate["interpretation"]) > 20, "Interpretation field must be informative"
    # Must contain the representation-level framing
    interp = gate["interpretation"].lower()
    assert "representation" in interp, "Interpretation must use representation-level language"
    # Must contain a negation of causal claims (not a positive causal claim)
    assert "not evidence" in interp or "not causal" in interp or "not claim" in interp, (
        "Interpretation must explicitly disclaim causal interpretation"
    )
