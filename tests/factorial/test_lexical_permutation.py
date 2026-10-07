"""The lexical permutation keeps unigram counts, lengths and POS classes, and Dp shuffles Cp."""

from __future__ import annotations

import json
import sys
from collections import Counter

from brain_code.training.corpus_transforms import _WORD_RE, is_content_word
from scripts import build_lexical_permutation

POS_MAP = {
    "cat": "NOUN",
    "dog": "NOUN",
    "ball": "NOUN",
    "ran": "VERB",
    "saw": "VERB",
    "big": "ADJ",
    "red": "ADJ",
    "slowly": "ADV"
}
TEXTS = [
    "The big cat ran. A red dog saw the ball!",
    "Cat saw dog slowly.",
    "The ball was red and big.",
    "A dog ran to the cat."
]


def _content_by_class(text: str) -> list[tuple[str, str]]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return [(POS_MAP.get(w, "UNK"), w) for w in words if is_content_word(w)]


def test_cells_preserve_counts_classes_and_lengths(tmp_path, monkeypatch):
    (tmp_path / "condition_A.txt").write_text("\n".join(TEXTS) + "\n")
    pos_file = tmp_path / "pos_map.json"
    pos_file.write_text(json.dumps({"pos_map": POS_MAP}))
    argv = [
        "build_lexical_permutation",
        "--corpora",
        str(tmp_path),
        "--pos-map",
        str(pos_file)
    ]
    monkeypatch.setattr(sys, "argv", argv)
    build_lexical_permutation.main()

    cp = (tmp_path / "condition_Cp.txt").read_text().splitlines()
    dp = (tmp_path / "condition_Dp.txt").read_text().splitlines()
    assert len(cp) == len(dp) == len(TEXTS)
    source_words = Counter(w for t in TEXTS for _, w in _content_by_class(t))
    permuted_words = Counter(w for t in cp for _, w in _content_by_class(t))
    assert source_words == permuted_words
    for original, permuted, shuffled in zip(TEXTS, cp, dp):
        assert [c for c, _ in _content_by_class(original)] == [c for c, _ in _content_by_class(permuted)]
        assert len(original.split()) == len(permuted.split())
        assert sorted(permuted.split()) == sorted(shuffled.split())
    meta = json.loads((tmp_path / "meta_lexical.json").read_text())
    assert meta["exact_unigram_match_all_classes"]
