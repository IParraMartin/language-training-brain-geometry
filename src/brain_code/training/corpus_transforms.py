"""Auditable corpus transformations for the factorial text manipulations.

The conditions are named after the operations they actually perform. They are
not described as removing syntax or semantics: each manipulation changes
multiple linguistic and tokenisation properties and must be accompanied by the
balance diagnostics produced by the corpus builder.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_VOWELS = tuple("aeiou")
_CONSONANTS = tuple("bcdfghjklmnprstvwxyz")
_FUNCTION_WORDS: frozenset[str] = frozenset({
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "can",
    "could",
    "must",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "up",
    "about",
    "into",
    "through",
    "during",
    "and",
    "but",
    "or",
    "nor",
    "not",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "this",
    "that",
    "these",
    "those",
    "as",
    "if",
    "then",
    "else",
    "while",
    "return",
    "def",
    "class",
    "import",
    "pass",
    "break",
    "continue",
    "lambda",
    "yield",
    "del",
    "raise",
    "try",
    "except",
    "finally",
    "with",
    "assert",
    "global",
    "nonlocal",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "how",
    "why",
    "there",
    "here",
    "than"
})
_WORD_RE = re.compile(r"\b([a-zA-Z]+)\b")
_SENT_END = re.compile(r"(?<=[.!?])\s+")

CONDITION_NAMES = {
    "A": "intact_control",
    "B": "within_sentence_whitespace_token_shuffle",
    "C": "type_consistent_pseudolexical_substitution",
    "D": "pseudolexical_substitution_plus_within_sentence_shuffle"
}
CONDITION_LABELS = tuple(CONDITION_NAMES)


def split_sentences(text: str) -> list[str]:
    """Splits text on whitespace that follows sentence-final punctuation."""
    parts = _SENT_END.split(text.strip())
    return [part for part in parts if part.strip()]


def is_content_word(word: str) -> bool:
    """Returns whether a word, stripped of edge punctuation, is alphabetic and not a function word."""
    core = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", word).lower()
    lookup = core.rstrip("s") if core.endswith("s") and len(core) > 2 else core
    return bool(core) and core.isalpha() and core not in _FUNCTION_WORDS and lookup not in _FUNCTION_WORDS


def _hash_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _candidate_pseudoword(word: str, attempt: int, seed: int) -> str:
    """Generate a deterministic candidate without claiming phonological validity."""
    digest = hashlib.sha256(f"{seed}\0{word.lower()}\0{attempt}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    # keep the source length while the candidate space allows; once it is exhausted for short words,
    # lengthen the candidate rather than emit a duplicate mapping
    length = len(word) if attempt < 4096 else len(word) + 1 + (attempt - 4096) // 4096
    length = max(length, 2)
    chars: list[str] = []
    phase = int(rng.integers(0, 2))
    for index in range(length):
        pool = _CONSONANTS if (index + phase) % 2 == 0 else _VOWELS
        chars.append(str(rng.choice(pool)))
    return "".join(chars)


@dataclass(frozen=True)
class PseudowordAudit:
    """Injectivity audit of a pseudoword codebook."""

    source_vocabulary_size: int
    mapping_size: int
    unique_output_size: int
    real_vocabulary_collisions: tuple[str, ...]
    duplicate_output_collisions: tuple[str, ...]
    identity_mappings: tuple[str, ...]
    mapping_hash: str

    @property
    def is_injective(self) -> bool:
        return (
            self.mapping_size == self.unique_output_size
            and not self.real_vocabulary_collisions
            and not self.duplicate_output_collisions
            and not self.identity_mappings
        )


class SentenceShuffler:
    """Shuffle whitespace-delimited tokens separately within each sentence."""

    operation_name = CONDITION_NAMES["B"]

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

    @staticmethod
    def _transform_one(text: str, rng: np.random.Generator) -> str:
        sentences = split_sentences(text)
        if not sentences:
            words = text.split()
            rng.shuffle(words)
            return " ".join(words)
        transformed: list[str] = []
        for sentence in sentences:
            words = sentence.split()
            rng.shuffle(words)
            transformed.append(" ".join(words))
        return " ".join(transformed)

    def transform_corpus(self, texts: list[str]) -> list[str]:
        rng = np.random.default_rng(self.seed)
        return [self._transform_one(text, rng) for text in texts]


class PseudowordSubstitutor:
    """Replace content-word types with an injective deterministic codebook."""

    operation_name = CONDITION_NAMES["C"]

    def __init__(
        self,
        *,
        seed: int = 0,
        forbidden_tokens: Iterable[str] = (),
        max_attempts_per_word: int = 1_000_000
    ) -> None:
        self.seed = int(seed)
        self.forbidden_tokens = {str(token).lower() for token in forbidden_tokens}
        self.max_attempts_per_word = int(max_attempts_per_word)
        self._map: dict[str, str] = {}
        self._audit: PseudowordAudit | None = None

    def fit(self, texts: list[str]) -> PseudowordSubstitutor:
        """Builds the codebook from the content-word types of `texts`."""
        source_vocab = {
            word.lower()
            for text in texts
            for word in _WORD_RE.findall(text)
        }
        content_vocab = sorted(word for word in source_vocab if is_content_word(word))
        forbidden = set(source_vocab) | self.forbidden_tokens | set(_FUNCTION_WORDS)
        used: set[str] = set()
        mapping: dict[str, str] = {}
        for word in content_vocab:
            for attempt in range(self.max_attempts_per_word):
                candidate = _candidate_pseudoword(word, attempt, self.seed).lower()
                if candidate == word or candidate in forbidden or candidate in used:
                    continue
                mapping[word] = candidate
                used.add(candidate)
                break
            else:
                raise RuntimeError(
                    f"Could not create a unique pseudoword for {word!r} after "
                    f"{self.max_attempts_per_word} attempts."
                )
        self._map = mapping
        self._audit = self._make_audit(source_vocab)
        if not self._audit.is_injective:
            raise AssertionError(f"Internal pseudoword-map validation failed: {self._audit}")
        return self

    def _make_audit(self, source_vocab: set[str]) -> PseudowordAudit:
        values = list(self._map.values())
        duplicates = sorted({value for value in values if values.count(value) > 1})
        real_collisions = sorted(set(values) & source_vocab)
        identities = sorted(word for word, pseudo in self._map.items() if word == pseudo)
        return PseudowordAudit(
            source_vocabulary_size=len(source_vocab),
            mapping_size=len(self._map),
            unique_output_size=len(set(values)),
            real_vocabulary_collisions=tuple(real_collisions),
            duplicate_output_collisions=tuple(duplicates),
            identity_mappings=tuple(identities),
            mapping_hash=_hash_json(self._map)
        )

    @property
    def vocabulary_size(self) -> int:
        return len(self._map)

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._map)

    @property
    def audit(self) -> PseudowordAudit:
        if self._audit is None:
            raise RuntimeError("Call fit() before reading the audit.")
        return self._audit

    def save_mapping(self, path: str | Path) -> None:
        if self._audit is None:
            raise RuntimeError("Call fit() before saving the mapping.")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "operation": self.operation_name,
            "seed": self.seed,
            "mapping": self._map,
            "audit": {
                **self._audit.__dict__,
                "real_vocabulary_collisions": list(self._audit.real_vocabulary_collisions),
                "duplicate_output_collisions": list(self._audit.duplicate_output_collisions),
                "identity_mappings": list(self._audit.identity_mappings)
            }
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _transform_one(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            word = match.group(1)
            pseudo = self._map.get(word.lower())
            if pseudo is None:
                return word
            if word.isupper():
                return pseudo.upper()
            if word[0].isupper():
                return pseudo[0].upper() + pseudo[1:]
            return pseudo

        return _WORD_RE.sub(replace, text)

    def transform_corpus(self, texts: list[str]) -> list[str]:
        if self._audit is None:
            raise RuntimeError("Call fit() before transform_corpus().")
        return [self._transform_one(text) for text in texts]


class BothTransforms:
    """Pseudoword substitution followed by within-sentence shuffling."""

    operation_name = CONDITION_NAMES["D"]

    def __init__(self, seed: int = 42, *, forbidden_tokens: Iterable[str] = ()) -> None:
        self.seed = int(seed)
        self.forbidden_tokens = tuple(forbidden_tokens)
        self.substitutor: PseudowordSubstitutor | None = None

    def transform_corpus(self, texts: list[str]) -> list[str]:
        self.substitutor = PseudowordSubstitutor(seed=self.seed, forbidden_tokens=self.forbidden_tokens).fit(texts)
        substituted = self.substitutor.transform_corpus(texts)
        return SentenceShuffler(seed=self.seed).transform_corpus(substituted)


def apply_condition(
    condition: str,
    texts: list[str],
    seed: int = 42,
    *,
    forbidden_tokens: Iterable[str] = ()
) -> list[str]:
    """Apply A/B/C/D while exposing operation-accurate names in metadata."""
    condition = condition.upper()
    if condition == "A":
        return list(texts)
    if condition == "B":
        return SentenceShuffler(seed=seed).transform_corpus(list(texts))
    if condition == "C":
        substitutor = PseudowordSubstitutor(seed=seed, forbidden_tokens=forbidden_tokens).fit(list(texts))
        return substitutor.transform_corpus(list(texts))
    if condition == "D":
        return BothTransforms(seed=seed, forbidden_tokens=forbidden_tokens).transform_corpus(list(texts))
    raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITION_LABELS}.")
