"""Builds the lexically permuted factorial cells Cp and Dp from the intact condition A corpus.

Cp redraws every content word from a shuffled permutation of its POS class's occurrences, keeping lengths,
POS sequence and exact unigram counts; Dp then shuffles words within sentences. Reads condition_A.txt in
--corpora and writes condition_Cp.txt, condition_Dp.txt and meta_lexical.json; fails on any count change.

Example: python -m scripts.build_lexical_permutation --corpora data/factorial_corpora_500m
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from brain_code.config import ROOT
from brain_code.training.corpus_transforms import _WORD_RE, SentenceShuffler, is_content_word

UNK_CLASS = "UNK"
CHUNK = 20_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpora", type=Path, required=True)
    p.add_argument("--pos-map", type=Path, default=ROOT / "data" / "factorial" / "pos_map.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@lru_cache(maxsize=None)
def content_word(word: str) -> bool:
    # about 4e8 occurrences over about 5e4 forms, so memoising the pure predicate dominates the runtime
    return is_content_word(word)


def match_case(source: str, replacement: str) -> str:
    """Gives the replacement the source word's capitalisation."""
    if source.isupper():
        return replacement.upper()
    if source[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(
    chunk: list[str],
    pos_map: dict[str, str],
    state: dict,
    rng_dp: np.random.Generator,
    out_cp,
    out_dp
) -> int:
    """Writes one chunk of texts to both outputs, consuming each class's pre-shuffled occurrence pool."""
    vocab, pool, cursor = state["vocab"], state["pool"], state["cursor"]
    realised, identity = state["realised"], state["identity"]
    per_class: dict[str, int] = {}
    parsed = []
    for line in chunk:
        spans = []
        for match in _WORD_RE.finditer(line):
            low = match.group(1).lower()
            if not content_word(low):
                continue
            cls = pos_map.get(low, UNK_CLASS)
            spans.append((
                match.start(1),
                match.end(1),
                cls,
                match.group(1)
            ))
            per_class[cls] = per_class.get(cls, 0) + 1
        parsed.append((line, spans))

    draws: dict[str, np.ndarray] = {}
    for cls, n in per_class.items():
        start = cursor[cls]
        draws[cls] = vocab[cls][pool[cls][start:start + n]]
        cursor[cls] = start + n
    local = {cls: 0 for cls in draws}

    for line, spans in parsed:
        if spans:
            pieces, prev = [], 0
            for start, end, cls, original in spans:
                pieces.append(line[prev:start])
                pick = draws[cls][local[cls]]
                local[cls] += 1
                bucket = realised.setdefault(cls, {})
                bucket[pick] = bucket.get(pick, 0) + 1
                identity["total"] += 1
                identity["same"] += pick == original.lower()
                pieces.append(match_case(original, pick))
                prev = end
            pieces.append(line[prev:])
            text_cp = "".join(pieces)
        else:
            text_cp = line
        out_cp.write(text_cp)
        out_dp.write(SentenceShuffler._transform_one(text_cp.rstrip("\n"), rng_dp) + "\n")
    return len(chunk)


def main() -> None:
    args = parse_args()
    source = args.corpora / "condition_A.txt"
    out_cp = args.corpora / "condition_Cp.txt"
    out_dp = args.corpora / "condition_Dp.txt"
    for path in (source, args.pos_map):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    pos_map: dict[str, str] = json.loads(args.pos_map.read_text())["pos_map"]
    # an empty or near-degenerate class inventory means the tagger did not run, and Cp would then swap
    # nouns into verb slots, destroying syntax as well
    classes = set(pos_map.values())
    if any(not c or not c.strip() for c in classes) or len(classes) < 4:
        raise SystemExit(f"degenerate POS classes in {args.pos_map}: {sorted(classes)!r}")

    # pass 1: occurrence counts of every content word, per POS class
    counts: dict[str, dict[str, int]] = {}
    n_texts = 0
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            n_texts += 1
            for word in _WORD_RE.findall(line):
                low = word.lower()
                if not content_word(low):
                    continue
                bucket = counts.setdefault(pos_map.get(low, UNK_CLASS), {})
                bucket[low] = bucket.get(low, 0) + 1
    print(f"{n_texts:,} texts, {sum(sum(b.values()) for b in counts.values()):,} content-word occurrences", flush=True)

    # an exact shuffled permutation of each class's occurrence multiset keeps unigram counts exact by
    # construction, which i.i.d. resampling would match only in expectation
    rng_pool = np.random.default_rng(args.seed)
    state: dict = {
        "vocab": {},
        "pool": {},
        "cursor": {},
        "realised": {},
        "identity": {
            "same": 0,
            "total": 0
        }
    }
    for cls, bucket in counts.items():
        words = sorted(bucket)
        state["vocab"][cls] = np.array(words, dtype=object)
        occurrences = np.repeat(
            np.arange(len(words), dtype=np.int32),
            np.array([bucket[w] for w in words], dtype=np.int64)
        )
        rng_pool.shuffle(occurrences)
        state["pool"][cls] = occurrences
        state["cursor"][cls] = 0

    # pass 2: write Cp and its within-sentence shuffle Dp
    rng_dp = np.random.default_rng(args.seed)
    done = 0
    with source.open(encoding="utf-8", errors="replace") as handle, \
            out_cp.open("w", encoding="utf-8") as cp_handle, \
            out_dp.open("w", encoding="utf-8") as dp_handle:
        chunk: list[str] = []
        for line in handle:
            chunk.append(line)
            if len(chunk) == CHUNK:
                done += emit(
                    chunk,
                    pos_map,
                    state,
                    rng_dp,
                    cp_handle,
                    dp_handle
                )
                chunk = []
        if chunk:
            done += emit(
                chunk,
                pos_map,
                state,
                rng_dp,
                cp_handle,
                dp_handle
            )
    if done != n_texts:
        raise SystemExit(f"wrote {done:,} texts, source had {n_texts:,}")

    # pass 3: the realised unigram counts must equal the source's exactly
    report = {}
    for cls in sorted(counts):
        want, got = counts[cls], state["realised"].get(cls, {})
        n_want, n_got = sum(want.values()), sum(got.values())
        report[cls] = {
            "exact_unigram_match": want == got,
            "total_variation_distance": 0.5 * sum(abs(got.get(w, 0) / max(n_got, 1) - want[w] / n_want) for w in want),
            "occurrences_intended": n_want,
            "occurrences_realised": n_got
        }
    identity = state["identity"]
    meta = {
        "built_from": str(source),
        "source_sha256": sha256_file(source),
        "seed": args.seed,
        "pos_map_types": len(pos_map),
        "n_texts": n_texts,
        "conditions": {
            "Cp": {
                "operation": "within_pos_class_occurrence_resampling",
                "sha256": sha256_file(out_cp),
                "n_texts": n_texts
            },
            "Dp": {
                "operation": "within_pos_class_occurrence_resampling_plus_within_sentence_shuffle",
                "sha256": sha256_file(out_dp),
                "n_texts": n_texts
            }
        },
        "unigram_fidelity_per_class": report,
        "exact_unigram_match_all_classes": all(r["exact_unigram_match"] for r in report.values()),
        "residual_identity_rate": identity["same"] / max(identity["total"], 1),
        "residual_identity_counts": identity
    }
    (args.corpora / "meta_lexical.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    if not meta["exact_unigram_match_all_classes"]:
        raise SystemExit("unigram counts not exactly preserved")
    print(f"wrote {out_cp} and {out_dp}")


if __name__ == "__main__":
    main()
