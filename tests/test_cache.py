"""Tests for content-addressed cache key utilities."""

from __future__ import annotations

from brain_code.utils.cache_utils import make_cache_key, keyed_cache_path


def test_same_inputs_produce_same_key():
    k1 = make_cache_key(corpus="python", tokenizer="/data/50k", text_field="code")
    k2 = make_cache_key(corpus="python", tokenizer="/data/50k", text_field="code")
    assert k1 == k2


def test_different_corpus_produces_different_key():
    k1 = make_cache_key(corpus="python", tokenizer="/data/50k")
    k2 = make_cache_key(corpus="assembly", tokenizer="/data/50k")
    assert k1 != k2


def test_different_tokenizer_produces_different_key():
    k1 = make_cache_key(corpus="python", tokenizer="/data/50k")
    k2 = make_cache_key(corpus="python", tokenizer="/data/32k")
    assert k1 != k2


def test_different_text_field_produces_different_key():
    k1 = make_cache_key(corpus="python", tokenizer="/data/50k", field="code")
    k2 = make_cache_key(corpus="python", tokenizer="/data/50k", field="text")
    assert k1 != k2


def test_key_is_16_hex_characters():
    k = make_cache_key(a=1, b=2)
    assert len(k) == 16
    assert all(c in "0123456789abcdef" for c in k)


def test_keyed_cache_path_changes_when_field_changes(tmp_path):
    p1 = keyed_cache_path(
        tmp_path,
        "freq",
        ".pkl",
        corpus="python",
        tok="/data/50k"
    )
    p2 = keyed_cache_path(
        tmp_path,
        "freq",
        ".pkl",
        corpus="assembly",
        tok="/data/50k"
    )
    assert p1 != p2
    assert p1.parent == tmp_path
    assert p1.name.startswith("freq_")
    assert p1.suffix == ".pkl"
