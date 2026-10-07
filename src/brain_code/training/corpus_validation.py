"""Manipulation checks for the factorial corpus conditions.

A corpus that fails its intended manipulation check must be redesigned before any training run. Every
condition is compared with A on unigram frequencies and sequence length; B and D must also raise bigram
entropy, and C and D must replace most content-word types.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from brain_code.training.corpus_transforms import _FUNCTION_WORDS, is_content_word

_WORD_RE = re.compile(r"\b([a-zA-Z]+)\b")


def _unigram_counts(texts: list[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(w.lower() for w in text.split())
    return counts


def _bigram_entropy(texts: list[str]) -> float:
    """Shannon entropy of word bigram distribution (bits)."""
    bigrams: Counter = Counter()
    for text in texts:
        words = text.lower().split()
        bigrams.update(zip(words, words[1:]))
    total = sum(bigrams.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in bigrams.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs + 1e-15)))


def _length_distribution(texts: list[str]) -> np.ndarray:
    """Array of word counts per text."""
    return np.array([len(t.split()) for t in texts], dtype=np.float64)


def _content_word_types(texts: list[str]) -> set[str]:
    types: set[str] = set()
    for text in texts:
        for w in _WORD_RE.findall(text):
            if is_content_word(w):
                types.add(w.lower())
    return types


@dataclass
class ValidationResult:
    condition: str
    check: str
    passed: bool
    metrics: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] Condition {self.condition} / {self.check}"]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        for f in self.failures:
            lines.append(f"  FAIL: {f}")
        return "\n".join(lines)


def check_unigram_match(
    texts_a: list[str],
    texts_x: list[str],
    condition: str,
    top_k: int = 100,
    rtol: float = 0.25,
    function_words_only: bool = False
) -> ValidationResult:
    """Top-k unigram frequencies should not change more than rtol (relative).

    For conditions C/D (pseudoword substitution), set function_words_only=True
    to restrict comparison to function words, which C/D must not alter.
    """
    counts_a = _unigram_counts(texts_a)
    counts_x = _unigram_counts(texts_x)
    total_a = max(sum(counts_a.values()), 1)
    total_x = max(sum(counts_x.values()), 1)
    if function_words_only:
        top_types = [w for w in _FUNCTION_WORDS if counts_a.get(w, 0) > 0]
    else:
        top_types = [w for w, _ in counts_a.most_common(top_k)]
    diffs = [
        abs(counts_x.get(w, 0) / total_x - counts_a[w] / total_a) / (counts_a[w] / total_a)
        for w in top_types if counts_a[w] > 0
    ]
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    failures = []
    if mean_diff > rtol:
        failures.append(f"Mean relative unigram frequency drift {mean_diff:.3f} > tolerance {rtol}")
    check_name = "function_word_unigram_match" if function_words_only else "unigram_match"
    return ValidationResult(
        condition=condition,
        check=check_name,
        passed=not failures,
        metrics={
            "mean_relative_unigram_diff": mean_diff,
            "n_types_checked": len(top_types),
            "function_words_only": function_words_only
        },
        failures=failures
    )


def check_length_match(
    texts_a: list[str],
    texts_x: list[str],
    condition: str,
    rtol: float = 0.05
) -> ValidationResult:
    """Mean sequence length should not change by more than rtol (relative)."""
    mean_a = float(_length_distribution(texts_a).mean())
    mean_x = float(_length_distribution(texts_x).mean())
    rel_diff = abs(mean_x - mean_a) / max(mean_a, 1)
    failures = []
    if rel_diff > rtol:
        failures.append(
            f"Mean length changed from {mean_a:.1f} to {mean_x:.1f} "
            f"(relative diff {rel_diff:.3f} > {rtol})"
        )
    return ValidationResult(
        condition=condition,
        check="length_match",
        passed=not failures,
        metrics={"mean_length_a": mean_a, "mean_length_x": mean_x, "rel_diff": rel_diff},
        failures=failures
    )


def check_bigram_entropy_increases(
    texts_a: list[str],
    texts_b: list[str],
    condition: str = "B"
) -> ValidationResult:
    """Shuffled corpus (B/D) should have higher bigram entropy than A."""
    h_a = _bigram_entropy(texts_a)
    h_b = _bigram_entropy(texts_b)
    failures = []
    if h_b <= h_a:
        failures.append(
            f"Bigram entropy of {condition} ({h_b:.4f} bits) "
            f"should exceed A ({h_a:.4f} bits)"
        )
    return ValidationResult(
        condition=condition,
        check="bigram_entropy_increases",
        passed=not failures,
        metrics={"bigram_entropy_a": h_a, f"bigram_entropy_{condition.lower()}": h_b},
        failures=failures
    )


def check_content_types_changed(
    texts_a: list[str],
    texts_c: list[str],
    condition: str = "C",
    min_changed_frac: float = 0.5
) -> ValidationResult:
    """Condition C/D should replace the majority of content word types."""
    types_a = _content_word_types(texts_a)
    types_c = _content_word_types(texts_c)
    if not types_a:
        return ValidationResult(
            condition=condition,
            check="content_types_changed",
            passed=True,
            metrics={"types_a": 0, "types_c": len(types_c), "changed_frac": 1.0}
        )
    overlap = types_a & types_c
    changed_frac = 1.0 - len(overlap) / len(types_a)
    failures = []
    if changed_frac < min_changed_frac:
        failures.append(
            f"Only {changed_frac:.1%} of content word types changed "
            f"(minimum required: {min_changed_frac:.1%})"
        )
    return ValidationResult(
        condition=condition,
        check="content_types_changed",
        passed=not failures,
        metrics={
            "types_a": len(types_a),
            "types_c": len(types_c),
            "overlap": len(overlap),
            "changed_frac": changed_frac
        },
        failures=failures
    )


def run_validation_battery(
    texts_a: list[str],
    texts_b: list[str] | None = None,
    texts_c: list[str] | None = None,
    texts_d: list[str] | None = None,
    verbose: bool = True
) -> dict[str, list[ValidationResult]]:
    """Runs every manipulation check for the conditions provided.

    Args:
        texts_a: The intact control corpus.
        texts_b: Shuffled corpus, if any.
        texts_c: Pseudoword corpus, if any.
        texts_d: Pseudoword plus shuffled corpus, if any.
        verbose: Print each result and an overall verdict.

    Returns:
        The ValidationResult list for each condition label.
    """
    results: dict[str, list[ValidationResult]] = {}
    pairs = [("B", texts_b), ("C", texts_c), ("D", texts_d)]

    for label, texts_x in pairs:
        if texts_x is None:
            continue
        # pseudoword substitution replaces content words on purpose, so only function words are compared
        func_only = label in ("C", "D")
        checks = [
            check_unigram_match(
                texts_a,
                texts_x,
                label,
                function_words_only=func_only
            ),
            check_length_match(texts_a, texts_x, label)
        ]
        if label in ("B", "D"):
            checks.append(check_bigram_entropy_increases(texts_a, texts_x, label))
        if label in ("C", "D"):
            checks.append(check_content_types_changed(texts_a, texts_x, label))
        results[label] = checks

    if verbose:
        for checks in results.values():
            for r in checks:
                print(r)
        status = "ALL CHECKS PASSED" if all_passed(results) else "SOME CHECKS FAILED — redesign corpus"
        print(f"\n{'='*60}\n{status}\n{'='*60}")

    return results


def all_passed(results: dict[str, list[ValidationResult]]) -> bool:
    """Returns whether every check in the battery passed."""
    return all(r.passed for checks in results.values() for r in checks)
