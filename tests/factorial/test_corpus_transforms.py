"""Tests for the factorial corpus transformations."""

from __future__ import annotations

import re

from brain_code.training.corpus_transforms import (
    apply_condition,
    is_content_word,
    split_sentences,
    PseudowordSubstitutor,
    _candidate_pseudoword
)


_SAMPLE = [
    "The cat sat on the mat and looked around.",
    "Scientists discovered a new species of fish in the deep ocean.",
    "She quickly ran through the forest to find shelter from the rain.",
    "The ancient ruins revealed many secrets about the lost civilization.",
    "He carefully placed the fragile vase on the wooden shelf."
]

_WORD_RE = re.compile(r"\b([a-zA-Z]+)\b")


def test_split_sentences_basic():
    sents = split_sentences("Hello world. How are you? I am fine!")
    assert len(sents) == 3


def test_split_sentences_single():
    sents = split_sentences("No punctuation here")
    assert len(sents) == 1
    assert sents[0] == "No punctuation here"


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_function_words_are_not_content():
    for w in ["the", "a", "is", "was", "for", "and", "or", "not", "in", "if"]:
        assert not is_content_word(w), f"'{w}' should not be a content word"


def test_nouns_are_content():
    for w in ["cat", "elephant", "civilization", "scientist", "forest"]:
        assert is_content_word(w), f"'{w}' should be a content word"


def test_empty_is_not_content():
    assert not is_content_word("")
    assert not is_content_word("123")


def test_pseudoword_is_deterministic():
    assert _candidate_pseudoword("elephant", 0, 0) == _candidate_pseudoword("elephant", 0, 0)


def test_pseudoword_length_matched():
    for word in ["cat", "elephant", "civilization"]:
        pw = _candidate_pseudoword(word, 0, 0)
        assert len(pw) == len(word), f"pseudoword length mismatch for '{word}'"


def test_pseudoword_is_alphabetic():
    for word in ["cat", "elephant", "forest"]:
        pw = _candidate_pseudoword(word, 0, 0)
        assert pw.isalpha(), f"pseudoword should be alphabetic: '{pw}'"


def test_pseudoword_different_from_original():
    # A pseudoword should not equal the original (with high probability)
    words = ["cat", "elephant", "scientist", "civilization", "discovered"]
    for word in words:
        pw = _candidate_pseudoword(word, 0, 0)
        assert pw != word, f"pseudoword should differ from original: '{word}'"


def test_pseudoword_is_lower_when_input_lower():
    pw = _candidate_pseudoword("cat", 0, 0)
    assert pw == pw.lower()


def test_condition_a_is_identity():
    out = apply_condition("A", _SAMPLE)
    assert out == _SAMPLE


def test_condition_a_case_insensitive():
    out = apply_condition("a", _SAMPLE)
    assert out == _SAMPLE


def test_condition_b_same_words_different_order():
    out = apply_condition("B", _SAMPLE, seed=0)
    for orig, trans in zip(_SAMPLE, out):
        words_orig = sorted(_WORD_RE.findall(orig.lower()))
        words_trans = sorted(_WORD_RE.findall(trans.lower()))
        assert words_orig == words_trans, "Condition B should preserve bag of words"


def test_condition_b_not_identical_to_a():
    out = apply_condition("B", _SAMPLE, seed=0)
    # At least some texts should differ (shuffling on long enough sentences)
    assert any(o != t for o, t in zip(_SAMPLE, out)), (
        "Condition B should change at least some texts"
    )


def test_condition_b_deterministic():
    out1 = apply_condition("B", _SAMPLE, seed=42)
    out2 = apply_condition("B", _SAMPLE, seed=42)
    assert out1 == out2


def test_condition_b_seed_changes_output():
    out1 = apply_condition("B", _SAMPLE, seed=42)
    out2 = apply_condition("B", _SAMPLE, seed=99)
    assert out1 != out2


def test_sentence_shuffler_preserves_length():
    out = apply_condition("B", _SAMPLE, seed=0)
    for orig, trans in zip(_SAMPLE, out):
        assert len(orig.split()) == len(trans.split()), (
            "Condition B must not add or remove words"
        )


def test_condition_c_replaces_content_words():
    out = apply_condition("C", _SAMPLE)
    # Extract content word types from original and transformed
    orig_content = {
        w.lower() for text in _SAMPLE for w in _WORD_RE.findall(text) if is_content_word(w)
    }
    trans_content = {
        w.lower() for text in out for w in _WORD_RE.findall(text) if is_content_word(w)
    }
    # At least half of original content types should be gone
    overlap = orig_content & trans_content
    changed_frac = 1.0 - len(overlap) / max(len(orig_content), 1)
    assert changed_frac >= 0.5, (
        f"Condition C should replace most content words; only {changed_frac:.1%} changed"
    )


def test_condition_c_preserves_function_words():
    out = apply_condition("C", _SAMPLE)
    for orig, trans in zip(_SAMPLE, out):
        orig_lower = orig.lower()
        trans_lower = trans.lower()
        for fw in ["the", "and", "in", "on", "to", "of", "a"]:
            orig_count = orig_lower.split().count(fw)
            trans_count = trans_lower.split().count(fw)
            assert orig_count == trans_count, (
                f"Function word '{fw}' count changed in condition C: "
                f"{orig_count} → {trans_count}"
            )


def test_condition_c_preserves_word_count():
    out = apply_condition("C", _SAMPLE)
    for orig, trans in zip(_SAMPLE, out):
        assert len(orig.split()) == len(trans.split()), (
            "Condition C must not add or remove words"
        )


def test_condition_c_deterministic():
    out1 = apply_condition("C", _SAMPLE)
    out2 = apply_condition("C", _SAMPLE)
    assert out1 == out2


def test_pseudoword_substitutor_fit_required():
    import pytest
    sub = PseudowordSubstitutor()
    with pytest.raises(RuntimeError, match="fit"):
        sub.transform_corpus(_SAMPLE)


def test_condition_d_changes_content_and_order():
    out = apply_condition("D", _SAMPLE, seed=0)
    # Content words should differ from A (like C)
    orig_content = {
        w.lower() for text in _SAMPLE for w in _WORD_RE.findall(text) if is_content_word(w)
    }
    trans_content = {
        w.lower() for text in out for w in _WORD_RE.findall(text) if is_content_word(w)
    }
    overlap = orig_content & trans_content
    changed_frac = 1.0 - len(overlap) / max(len(orig_content), 1)
    assert changed_frac >= 0.5, "Condition D should replace most content words"


def test_condition_d_preserves_word_count():
    out = apply_condition("D", _SAMPLE, seed=0)
    for orig, trans in zip(_SAMPLE, out):
        assert len(orig.split()) == len(trans.split())


def test_condition_d_deterministic():
    out1 = apply_condition("D", _SAMPLE, seed=42)
    out2 = apply_condition("D", _SAMPLE, seed=42)
    assert out1 == out2


def test_unknown_condition_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown condition"):
        apply_condition("Z", _SAMPLE)
