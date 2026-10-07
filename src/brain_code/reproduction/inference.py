"""Seed-level inference for the corpus contrasts: weighted Welch contrasts, Holm correction, layer tests.

The replicate is an independently trained (or independently initialized) model. Cross-validation folds
never enter these tests as units.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import comb

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class Contrast:
    """A linear contrast between independent groups with its Welch-Satterthwaite test.

    Attributes:
        estimate: Weighted difference of group means.
        se: Standard error from the within-group sample variances.
        df: Welch-Satterthwaite degrees of freedom.
        t: estimate / se.
        p: Two-sided p-value.
        ci_low: Lower bound of the unadjusted 95% interval.
        ci_high: Upper bound of the unadjusted 95% interval.
    """

    estimate: float
    se: float
    df: float
    t: float
    p: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict:
        return asdict(self)


def welch_contrast(groups: list[tuple[float, NDArray]], offset: float = 0.0, level: float = 0.95) -> Contrast:
    """Tests sum_g w_g * mean(x_g) - offset with independent groups and unequal variances.

    With one group and no other variance source this is the one-sample t-test; with two groups weighted
    +1 and -1 it is Welch's t-test.

    Args:
        groups: ``(weight, observations)`` pairs; each group needs at least two observations.
        offset: Constant subtracted from the estimate. Treating a reference mean as fixed means passing
            it here instead of as a group, which ignores the reference's own sampling variance.
        level: Confidence level of the reported interval.

    Returns:
        The contrast estimate, its standard error, degrees of freedom, p-value and interval.
    """
    estimate = sum(w * float(np.mean(x)) for w, x in groups) - offset
    parts = [(w ** 2) * float(np.var(x, ddof=1)) / len(x) for w, x in groups]
    se = float(np.sqrt(sum(parts)))
    df = sum(parts) ** 2 / sum(part ** 2 / (len(x) - 1) for part, (_, x) in zip(parts, groups) if part > 0)
    t = estimate / se
    p = float(2 * stats.t.sf(abs(t), df))
    q = float(stats.t.ppf(0.5 + level / 2, df))
    return Contrast(
        estimate=float(estimate),
        se=se,
        df=float(df),
        t=float(t),
        p=p,
        ci_low=float(estimate - q * se),
        ci_high=float(estimate + q * se)
    )


def holm(p_values: list[float] | NDArray) -> NDArray:
    """Holm step-down adjusted p-values, monotone and capped at one."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def primary_contrasts(
    language: NDArray,
    code: dict[str, NDArray],
    random: NDArray,
    random_variance: bool = True
) -> dict[str, Contrast]:
    """The three prespecified contrasts on mean-over-depth scores.

    The code condition is the equally weighted mean of the four code corpora, each an independent group
    of seeds, so its variance is the weighted sum of within-corpus seed variances.

    Args:
        language: TinyStories depth summaries, one per seed.
        code: Depth summaries per code corpus, one per seed.
        random: Depth summaries of the random-weight initializations.
        random_variance: If true, the random initializations are a sampled group; if false, their mean
            is treated as a fixed reference.

    Returns:
        ``L-R``, ``C-R`` and ``L-C`` contrasts.
    """
    code_groups = [(0.25, np.asarray(x)) for x in code.values()]
    if len(code_groups) != 4:
        raise ValueError(f"expected four code corpora, got {sorted(code)}")
    if random_variance:
        reference, offset = [(-1.0, np.asarray(random))], 0.0
    else:
        reference, offset = [], float(np.mean(random))
    return {
        "L-R": welch_contrast([(1.0, np.asarray(language))] + reference, offset),
        "C-R": welch_contrast(code_groups + reference, offset),
        "L-C": welch_contrast([(1.0, np.asarray(language))] + [(-w, x) for w, x in code_groups])
    }


def exact_separation_p(a: NDArray, b: NDArray) -> float:
    """Two-sided exact rank-test p-value when every value of ``a`` exceeds every value of ``b``.

    Returns NaN when the samples overlap; for complete separation the Mann-Whitney statistic is extreme
    and p = 2 / C(n_a + n_b, n_a).
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.min() > b.max() or b.min() > a.max():
        return float(min(1.0, 2 / comb(len(a) + len(b), len(a))))
    return float("nan")


def layer_tests(language: NDArray, code: dict[str, NDArray]) -> dict[str, NDArray]:
    """Per-block TinyStories-versus-code and code-heterogeneity tests with Holm over blocks.

    The gap test is Welch's t between the TinyStories seeds and all code models pooled (20 models); the
    heterogeneity test is a one-way ANOVA across the four code corpora. Neither selects a block.

    Args:
        language: ``(n_seeds, n_blocks)`` TinyStories scores.
        code: Corpus -> ``(n_seeds, n_blocks)`` scores.

    Returns:
        Arrays over blocks: gap estimate, t, raw and Holm p for the gap; F, raw and Holm p for
        heterogeneity; and the leading code corpus.
    """
    language = np.asarray(language)
    pooled = np.concatenate([np.asarray(x) for x in code.values()], axis=0)
    n_blocks = language.shape[1]
    gap = language.mean(0) - np.mean([np.asarray(x).mean(0) for x in code.values()], axis=0)
    t_gap = np.empty(n_blocks)
    p_gap = np.empty(n_blocks)
    f_het = np.empty(n_blocks)
    p_het = np.empty(n_blocks)
    leaders = []
    names = list(code)
    for block in range(n_blocks):
        result = stats.ttest_ind(language[:, block], pooled[:, block], equal_var=False)
        t_gap[block], p_gap[block] = result.statistic, result.pvalue
        anova = stats.f_oneway(*[np.asarray(code[c])[:, block] for c in names])
        f_het[block], p_het[block] = anova.statistic, anova.pvalue
        leaders.append(names[int(np.argmax([np.asarray(code[c])[:, block].mean() for c in names]))])
    return {
        "gap": gap,
        "t_gap": t_gap,
        "p_gap": p_gap,
        "p_gap_holm": holm(p_gap),
        "f_heterogeneity": f_het,
        "p_heterogeneity": p_het,
        "p_heterogeneity_holm": holm(p_het),
        "leader": np.asarray(leaders)
    }


def order_agreement(means: dict[str, float], predicted: list[str]) -> tuple[float, bool]:
    """Spearman rho between observed corpus means and a predicted descending order, and exact match."""
    observed = np.asarray([means[c] for c in predicted])
    predicted_rank = np.arange(len(predicted), 0, -1)
    rho = float(stats.spearmanr(observed, predicted_rank).statistic)
    exact = list(np.argsort(-observed)) == list(range(len(predicted)))
    return rho, bool(exact)


def pairwise_mean(values: dict[tuple[str, str], float], members_a: list[str], members_b: list[str]) -> float:
    """Mean of a symmetric pairwise statistic over distinct pairs within one set or across two sets."""
    def lookup(x: str, y: str) -> float:
        return values[(x, y)] if (x, y) in values else values[(y, x)]

    if members_a == members_b:
        pairs = list(combinations(members_a, 2))
    else:
        pairs = [(x, y) for x in members_a for y in members_b]
    return float(np.mean([lookup(x, y) for x, y in pairs]))
