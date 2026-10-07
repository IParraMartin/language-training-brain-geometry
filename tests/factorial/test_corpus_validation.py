"""Tests for the factorial corpus validation suite."""

from __future__ import annotations

from brain_code.training.corpus_transforms import apply_condition
from brain_code.training.corpus_validation import (
    check_unigram_match,
    check_length_match,
    check_bigram_entropy_increases,
    check_content_types_changed,
    run_validation_battery,
    all_passed
)


_SAMPLE = [
    "The cat sat on the mat and looked around.",
    "Scientists discovered a new species of fish in the deep ocean.",
    "She quickly ran through the forest to find shelter from the rain.",
    "The ancient ruins revealed many secrets about the lost civilization.",
    "He carefully placed the fragile vase on the wooden shelf.",
    "Birds migrate south every winter to escape the cold.",
    "The researcher published important findings about climate change.",
    "Children learn language through exposure and regular practice.",
    "Mountains form over millions of years through geological processes.",
    "The programmer wrote elegant functions to parse the complex data.",
    "Rain fell heavily on the empty streets during the stormy night.",
    "They discovered that neurons communicate through chemical signals.",
    "The novel explores profound themes of memory and human identity.",
    "Engineers designed a strong bridge that could withstand earthquakes.",
    "Flowers bloom in spring when temperatures rise and days lengthen."
]


def test_unigram_match_passes_for_condition_b():
    texts_b = apply_condition("B", _SAMPLE, seed=42)
    result = check_unigram_match(_SAMPLE, texts_b, "B")
    assert result.passed, f"Unigram check failed for condition B: {result}"


def test_unigram_match_passes_for_condition_c_function_words_only():
    texts_c = apply_condition("C", _SAMPLE)
    # C substitutes content words; function word frequencies should be preserved.
    result = check_unigram_match(
        _SAMPLE,
        texts_c,
        "C",
        function_words_only=True
    )
    assert result.passed, str(result)


def test_unigram_match_passes_for_identical_corpus():
    result = check_unigram_match(_SAMPLE, list(_SAMPLE), "A")
    assert result.passed


def test_unigram_match_fails_when_corpus_completely_different():
    dummy = ["xyz abc def ghi"] * len(_SAMPLE)
    result = check_unigram_match(
        _SAMPLE,
        dummy,
        "X",
        top_k=10,
        rtol=0.01
    )
    assert not result.passed


def test_length_match_passes_for_condition_b():
    texts_b = apply_condition("B", _SAMPLE, seed=42)
    result = check_length_match(_SAMPLE, texts_b, "B")
    assert result.passed, str(result)


def test_length_match_passes_for_condition_c():
    texts_c = apply_condition("C", _SAMPLE)
    result = check_length_match(_SAMPLE, texts_c, "C")
    assert result.passed, str(result)


def test_length_match_fails_when_texts_truncated():
    truncated = [t[:5] for t in _SAMPLE]
    result = check_length_match(
        _SAMPLE,
        truncated,
        "X",
        rtol=0.05
    )
    assert not result.passed


def test_bigram_entropy_increases_for_condition_b():
    texts_b = apply_condition("B", _SAMPLE, seed=0)
    result = check_bigram_entropy_increases(_SAMPLE, texts_b, "B")
    assert result.passed, str(result)


def test_bigram_entropy_unchanged_for_identical_fails():
    result = check_bigram_entropy_increases(_SAMPLE, list(_SAMPLE), "B")
    assert not result.passed


def test_content_types_changed_for_condition_c():
    texts_c = apply_condition("C", _SAMPLE)
    result = check_content_types_changed(_SAMPLE, texts_c, "C")
    assert result.passed, str(result)


def test_content_types_unchanged_for_identical_fails():
    result = check_content_types_changed(_SAMPLE, list(_SAMPLE), "C")
    assert not result.passed


def test_full_battery_passes_for_all_conditions():
    texts_b = apply_condition("B", _SAMPLE, seed=42)
    texts_c = apply_condition("C", _SAMPLE)
    texts_d = apply_condition("D", _SAMPLE, seed=42)
    results = run_validation_battery(
        texts_a=_SAMPLE,
        texts_b=texts_b,
        texts_c=texts_c,
        texts_d=texts_d,
        verbose=False
    )
    assert "B" in results and "C" in results and "D" in results
    assert all_passed(results), (
        "Full battery should pass for correctly generated conditions."
    )


def test_full_battery_fails_when_condition_b_is_unchanged():
    results = run_validation_battery(
        texts_a=_SAMPLE,
        texts_b=list(_SAMPLE),  # identical to A — bigram entropy won't increase
        texts_c=None,
        texts_d=None,
        verbose=False
    )
    assert not all_passed(results)


def test_full_battery_skips_none_conditions():
    results = run_validation_battery(
        texts_a=_SAMPLE,
        texts_b=None,
        texts_c=None,
        texts_d=None,
        verbose=False
    )
    assert results == {}


def test_all_passed_on_empty_results():
    assert all_passed({})
