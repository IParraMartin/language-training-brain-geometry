"""Seed-level contrasts, Holm correction and layer tests match their textbook definitions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from brain_code.reproduction.inference import (
    exact_separation_p,
    holm,
    layer_tests,
    primary_contrasts,
    welch_contrast
)


def test_one_group_contrast_is_the_one_sample_t_test():
    x = np.array([0.12, 0.13, 0.11, 0.125, 0.118])
    result = welch_contrast([(1.0, x)], offset=0.064)
    reference = stats.ttest_1samp(x, 0.064)
    assert result.t == pytest.approx(reference.statistic)
    assert result.p == pytest.approx(reference.pvalue)
    assert result.df == pytest.approx(4)


def test_two_group_contrast_is_welchs_t_test():
    a = np.array([0.12, 0.13, 0.11, 0.125, 0.118])
    b = np.array([0.05, 0.08, 0.06])
    result = welch_contrast([(1.0, a), (-1.0, b)])
    reference = stats.ttest_ind(a, b, equal_var=False)
    assert result.t == pytest.approx(reference.statistic)
    assert result.p == pytest.approx(reference.pvalue)


def test_interval_contains_estimate_and_scales_with_variance():
    rng = np.random.default_rng(0)
    narrow = welch_contrast([(1.0, rng.normal(1, 0.01, 5))])
    wide = welch_contrast([(1.0, rng.normal(1, 0.1, 5))])
    assert narrow.ci_low < narrow.estimate < narrow.ci_high
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_holm_matches_hand_computation():
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert np.all(adjusted <= 1)


def test_random_variance_widens_the_reference_contrasts_only():
    language = np.array([0.12, 0.12, 0.121, 0.131, 0.117])
    code = {c: np.array([
        0.06,
        0.07,
        0.065,
        0.068,
        0.07
    ]) + i * 0.001 for i, c in enumerate("abcd")}
    random = np.array([0.049, 0.064, 0.080])
    fixed = primary_contrasts(
        language,
        code,
        random,
        random_variance=False
    )
    sampled = primary_contrasts(
        language,
        code,
        random,
        random_variance=True
    )
    assert sampled["L-R"].estimate == pytest.approx(fixed["L-R"].estimate)
    assert sampled["L-R"].se > fixed["L-R"].se
    assert sampled["L-C"].p == pytest.approx(fixed["L-C"].p)


def test_exact_separation_p_for_five_versus_three():
    assert exact_separation_p([
        5,
        6,
        7,
        8,
        9
    ], [1, 2, 3]) == pytest.approx(2 / 56)
    assert np.isnan(exact_separation_p([1, 6, 7, 8, 9], [2, 3, 4]))


def test_layer_tests_detect_a_gap_without_selecting_a_layer():
    rng = np.random.default_rng(1)
    language = rng.normal(0.12, 0.005, (5, 12))
    language[:, 0] = rng.normal(0.08, 0.005, 5)
    code = {c: rng.normal(0.08, 0.005, (5, 12)) for c in "abcd"}
    result = layer_tests(language, code)
    assert result["p_gap_holm"][0] > 0.05
    assert np.all(result["p_gap_holm"][1:] < 0.05)
    assert len(result["leader"]) == 12
