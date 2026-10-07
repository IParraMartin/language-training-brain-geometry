"""Shared-tokenizer fairness diagnostics: stimulus coverage, vocabulary overlap and fertility."""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# hf_home has to be set before datasets and transformers are imported
_cache = str(Path(__file__).resolve().parents[3] / "cache")
os.makedirs(_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _cache)

from datasets import load_dataset
from transformers import AutoTokenizer

from brain_code.config import DATASETS, PATHS, EXPERIMENTS
from brain_code.diagnostics.measurement_tools import encode_without_specials


_RESULTS_DIR = Path("results") / "confounds"
_FREQ_CACHE_DIR = _RESULTS_DIR / "freq_tables"
_THRESHOLDS = [1, 10, 100, 1_000, 10_000]
_FREQ_OVERLAP_THRESHOLD = 100


def tokenize_stimuli(
    benchmark_identifier: str,
    tokenizer_name: str = PATHS["tokenizer"]["default"],
    out_path: Optional[Path] = None
) -> tuple[list[str], list[list[int]]]:
    """Tokenizes the stimulus sentences of a Brain-Score benchmark with the shared tokenizer.

    The result is pickled so later diagnostics reload it without fetching the
    benchmark again; an existing pickle is returned as is.

    Args:
        benchmark_identifier: Key from ``brain_code.config.EXPERIMENTS``
            (e.g. ``"pereira_243"``) or a full Brain-Score benchmark ID.
        tokenizer_name: Path or HuggingFace Hub ID of the tokenizer.
        out_path: Pickle path; defaults to
            ``results/confounds/stimulus_tokens.pkl``.

    Returns:
        ``(sentences, token_id_seqs)``: the raw stimuli in benchmark order and
        their token-ID sequences.
    """
    from brainscore_language import load_benchmark

    out_path = Path(out_path or (_RESULTS_DIR / "stimulus_tokens.pkl"))
    if out_path.exists():
        with open(out_path, "rb") as f:
            return pickle.load(f)

    full_id = EXPERIMENTS.get(benchmark_identifier, benchmark_identifier)
    print(f"Loading benchmark: {full_id}")
    benchmark = load_benchmark(full_id)
    sentences: list[str] = benchmark.data["stimulus"].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_id_seqs: list[list[int]] = [
        tokenizer(s, add_special_tokens=False)["input_ids"] for s in sentences
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump((sentences, token_id_seqs), f)

    print(f"Tokenized {len(sentences)} stimuli → {out_path}")
    return sentences, token_id_seqs


def build_corpus_freq_table(
    corpus_name: str,
    tokenizer_name: str = PATHS["tokenizer"]["default"],
    text_field: str = "code",
    batch_size: int = 4096,
    cache_dir: Optional[Path] = None
) -> Counter:
    """Counts every vocabulary ID over a full training corpus.

    Streams the corpus from HuggingFace and tokenizes it in batches. The counter
    is pickled to a content-addressed cache so later calls load it instantly.

    Args:
        corpus_name: Key from ``brain_code.config.DATASETS`` (e.g.
            ``"python"``).
        tokenizer_name: Path or HuggingFace Hub ID of the tokenizer.
        text_field: Dataset column holding the text.
        batch_size: Texts tokenized per call.
        cache_dir: Pickle cache directory; defaults to
            ``results/confounds/freq_tables/``.

    Returns:
        Vocabulary ID mapped to its training-corpus count.

    Raises:
        KeyError: If ``corpus_name`` is not in ``DATASETS``.
    """
    from brain_code.utils.cache_utils import keyed_cache_path
    cache_dir = Path(cache_dir or _FREQ_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # the key covers tokenizer and text field, so changing either invalidates the cache
    cache_path = keyed_cache_path(
        cache_dir,
        prefix=corpus_name,
        tokenizer_name=tokenizer_name,
        text_field=text_field
    )

    if cache_path.exists():
        print(f"Loading cached freq table: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    path, config = DATASETS[corpus_name]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    freq: Counter = Counter()
    batch: list[str] = []

    ds = load_dataset(
        path,
        config,
        split="train",
        streaming=True
    )
    for ex in tqdm(
        ds,
        desc=f"freq:{corpus_name}",
        ncols=100,
        leave=True
    ):
        batch.append(str(ex[text_field]))
        if len(batch) == batch_size:
            for seq in encode_without_specials(tokenizer, batch):
                freq.update(seq)
            batch = []

    if batch:
        for seq in encode_without_specials(tokenizer, batch):
            freq.update(seq)

    with open(cache_path, "wb") as f:
        pickle.dump(freq, f)
    print(f"Saved freq table ({len(freq):,} types) → {cache_path}")
    return freq


def compute_stimulus_coverage(
    stimulus_token_ids: list[list[int]],
    freq_tables: dict[str, Counter],
    thresholds: list[int] = _THRESHOLDS,
    out_path: Optional[Path] = None
) -> dict:
    """Computes stimulus token coverage per corpus at several frequency thresholds.

    Reports the fraction of unique stimulus token types seen in training at
    least ``T`` times, and the weighted token coverage (fraction of all token
    occurrences in the stimuli), which matters more for brain scoring because
    it reflects what the encoder actually processes.

    Args:
        stimulus_token_ids: Token-ID sequence of each stimulus, from
            :func:`tokenize_stimuli`.
        freq_tables: Per-corpus frequency tables from
            :func:`build_corpus_freq_table`.
        thresholds: Minimum training counts; a type is covered at ``T`` if its
            count is ``>= T``.
        out_path: JSON output path; defaults to
            ``results/confounds/coverage.json``.

    Returns:
        ``{corpus: {"type_cov_ge{T}": ..., "tok_cov_ge{T}": ...}}`` for every
        threshold ``T``.
    """
    out_path = Path(out_path or (_RESULTS_DIR / "coverage.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_occurrences: list[int] = [tok for seq in stimulus_token_ids for tok in seq]
    stimulus_types: set[int] = set(all_occurrences)
    n_types = len(stimulus_types)
    n_tokens = len(all_occurrences)

    results: dict[str, dict] = {}

    header = f"{'corpus':<15}" + "".join(
        f"  type≥{t:<6}  tok≥{t:<6}" for t in thresholds
    )
    print(header)
    print("-" * len(header))

    for corpus, freq in freq_tables.items():
        row: dict = {}
        row_str = f"{corpus:<15}"
        for t in thresholds:
            covered_types = sum(1 for tok in stimulus_types if freq.get(tok, 0) >= t)
            covered_toks = sum(1 for tok in all_occurrences if freq.get(tok, 0) >= t)
            type_frac = covered_types / n_types if n_types else 0.0
            tok_frac = covered_toks / n_tokens if n_tokens else 0.0
            row[f"type_cov_ge{t}"] = round(type_frac, 4)
            row[f"tok_cov_ge{t}"] = round(tok_frac, 4)
            row_str += f"  {type_frac:.4f}       {tok_frac:.4f}  "
        results[corpus] = row
        print(row_str)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved coverage table → {out_path}")
    return results


def compute_vocab_overlap(
    freq_tables: dict[str, Counter],
    stimulus_token_types: set[int],
    n_threshold: int = _FREQ_OVERLAP_THRESHOLD,
    out_path: Optional[Path] = None
) -> dict:
    """Computes pairwise Jaccard overlap of the corpus vocabularies.

    Each corpus has a "present" vocabulary (seen at least once) and a "robust"
    one (seen at least ``n_threshold`` times). The intersection of each with
    the stimulus token types is also reported, since that is the number the
    shared-tokenizer fairness argument rests on.

    Args:
        freq_tables: Per-corpus frequency tables from
            :func:`build_corpus_freq_table`.
        stimulus_token_types: Unique token IDs in the stimuli.
        n_threshold: Count threshold of the robust vocabulary.
        out_path: JSON output path; defaults to
            ``results/confounds/vocab_overlap.json``.

    Returns:
        A dict with the ``present`` and ``robust`` Jaccard matrices and
        ``stimulus_intersection`` (per-corpus intersection size and fraction
        of stimulus types, at both thresholds).
    """
    out_path = Path(out_path or (_RESULTS_DIR / "vocab_overlap.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _jaccard(a: set, b: set) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    corpus_names = list(freq_tables.keys())
    vocabs: dict[str, dict[str, set]] = {
        name: {
            "present": {tok for tok, cnt in freq.items() if cnt >= 1},
            "robust": {tok for tok, cnt in freq.items() if cnt >= n_threshold}
        }
        for name, freq in freq_tables.items()
    }

    results: dict = {"present": {}, "robust": {}, "stimulus_intersection": {}}

    for thresh_key, min_cnt in (("present", 1), ("robust", n_threshold)):
        matrix: dict[str, dict[str, float]] = {}
        for a in corpus_names:
            matrix[a] = {
                b: round(_jaccard(vocabs[a][thresh_key], vocabs[b][thresh_key]), 4)
                for b in corpus_names
            }
        results[thresh_key] = matrix

        print(f"\nJaccard overlap (threshold: {thresh_key}, count >= {min_cnt}):")
        header = f"{'':>12}" + "".join(f"{c:>12}" for c in corpus_names)
        print(header)
        for a in corpus_names:
            print(f"{a:>12}" + "".join(f"{matrix[a][b]:>12.4f}" for b in corpus_names))

    n_stim = len(stimulus_token_types)
    for name in corpus_names:
        entry: dict = {}
        for thresh_key in ("present", "robust"):
            isect = vocabs[name][thresh_key] & stimulus_token_types
            entry[thresh_key] = {
                "intersection_size": len(isect),
                "fraction_of_stimulus_types": round(len(isect) / n_stim if n_stim else 0.0, 4)
            }
        results["stimulus_intersection"][name] = entry

    print(f"\nStimulus-type intersection ({n_stim} unique types):")
    for name, entry in results["stimulus_intersection"].items():
        p = entry["present"]
        r = entry["robust"]
        print(
            f"  {name:<12}  present: {p['intersection_size']:>5} ({p['fraction_of_stimulus_types']:.4f})"
            f"  robust(>={n_threshold}): {r['intersection_size']:>5} ({r['fraction_of_stimulus_types']:.4f})"
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved vocab overlap → {out_path}")
    return results


def check_per_stimulus_fertility_variance(
    stimulus_sentences: list[str],
    stimulus_token_ids: list[list[int]],
    out_path: Optional[Path] = None
) -> dict:
    """Reports per-stimulus fertility (tokens per word) and flags outliers.

    A corpus-level average can hide stimuli that segment badly, so stimuli
    above the 95th percentile are flagged for inspection before aggregate
    fertility is used as a fairness argument.

    Args:
        stimulus_sentences: Raw stimulus strings.
        stimulus_token_ids: Token-ID sequences parallel to
            ``stimulus_sentences``.
        out_path: JSON output path; defaults to
            ``results/confounds/fertility_variance.json``.

    Returns:
        A dict with the mean, std, min, max and p95 of tokens per word,
        ``per_stimulus`` records and ``flagged_high_fertility`` indices.
    """
    out_path = Path(out_path or (_RESULTS_DIR / "fertility_variance.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_stimulus: list[dict] = []
    fertilities: list[float] = []

    for sentence, token_ids in zip(stimulus_sentences, stimulus_token_ids):
        n_words = len(sentence.split())
        n_tokens = len(token_ids)
        fertility = n_tokens / n_words if n_words else 0.0
        fertilities.append(fertility)
        per_stimulus.append({
            "sentence": sentence,
            "n_tokens": n_tokens,
            "n_words": n_words,
            "tokens_per_word": round(fertility, 4)
        })

    arr = np.array(fertilities, dtype=np.float64)
    p95 = float(np.percentile(arr, 95))
    flagged = [i for i, f in enumerate(fertilities) if f > p95]

    summary = {
        "mean_tokens_per_word": round(float(arr.mean()), 4),
        "std_tokens_per_word": round(float(arr.std()), 4),
        "min_tokens_per_word": round(float(arr.min()), 4),
        "max_tokens_per_word": round(float(arr.max()), 4),
        "p95_tokens_per_word": round(p95, 4),
        "per_stimulus": per_stimulus,
        "flagged_high_fertility": flagged
    }

    print(
        f"\nPer-stimulus fertility (tokens/word):\n"
        f"  mean={summary['mean_tokens_per_word']:.4f}  "
        f"std={summary['std_tokens_per_word']:.4f}  "
        f"min={summary['min_tokens_per_word']:.4f}  "
        f"max={summary['max_tokens_per_word']:.4f}  "
        f"p95={p95:.4f}\n"
        f"  Flagged {len(flagged)} stimuli above p95 threshold."
    )
    for idx in flagged[:5]:
        print(f"  [{idx}] ({fertilities[idx]:.2f} tok/word) {stimulus_sentences[idx]}")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved fertility variance → {out_path}")
    return summary


def run_all_diagnostics(
    benchmark_id: str = "pereira_243",
    tokenizer_name: str = PATHS["tokenizer"]["default"],
    corpora: Optional[list[str]] = None,
    batch_size: int = 4096,
    freq_table_cache_dir: Optional[Path] = None
) -> dict:
    """Runs stimulus tokenization, the frequency tables and diagnostics 1-3 in order.

    Embedding trainedness (diagnostic 4) runs per trained checkpoint, and
    diagnostics 5 and 6 need brain scores and trained models, so they are run
    separately through :func:`correlational_analysis` and
    :func:`run_embedding_ablation`.

    Args:
        benchmark_id: Key from ``brain_code.config.EXPERIMENTS`` or a full
            Brain-Score benchmark ID.
        tokenizer_name: Path or HuggingFace Hub ID of the shared tokenizer.
        corpora: Corpus names to process; ``None`` uses all of ``DATASETS``.
        batch_size: Tokenizer batch size for the frequency tables.
        freq_table_cache_dir: Frequency table cache directory; defaults to
            ``results/confounds/freq_tables/``.

    Returns:
        A dict with ``sentences``, ``token_id_seqs``, ``freq_tables``,
        ``coverage``, ``vocab_overlap`` and ``fertility_variance``.
    """
    corpora = corpora or list(DATASETS.keys())

    print("=" * 70)
    print("SHARED-TOKENIZER FAIRNESS DIAGNOSTICS")
    print("=" * 70)

    print("\n[Step 1] Tokenizing stimuli...")
    sentences, token_id_seqs = tokenize_stimuli(benchmark_id, tokenizer_name)
    stimulus_token_types: set[int] = {tok for seq in token_id_seqs for tok in seq}
    print(
        f"  {len(sentences)} stimuli  |  "
        f"{sum(len(s) for s in token_id_seqs)} total tokens  |  "
        f"{len(stimulus_token_types)} unique token types"
    )

    print("\n[Step 2] Building corpus frequency tables...")
    freq_tables: dict[str, Counter] = {}
    for corpus in corpora:
        print(f"  → {corpus}")
        freq_tables[corpus] = build_corpus_freq_table(
            corpus_name=corpus,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            cache_dir=freq_table_cache_dir
        )

    print("\n[Diagnostic 1] Stimulus token coverage:")
    coverage = compute_stimulus_coverage(token_id_seqs, freq_tables)

    print("\n[Diagnostic 2] Cross-corpus vocabulary overlap:")
    vocab_overlap = compute_vocab_overlap(freq_tables, stimulus_token_types)

    print("\n[Small-things] Per-stimulus fertility variance:")
    fertility_variance = check_per_stimulus_fertility_variance(sentences, token_id_seqs)

    print("\n" + "=" * 70)
    print("Diagnostics 1-2 complete. Remaining steps:")
    print("compute_embedding_trainedness() per model (diagnostic 3)")
    print("correlational_analysis() once brain scores exist (diagnostic 4)")
    print("run_embedding_ablation() on assembly or binary (diagnostic 5)")
    print("=" * 70)

    return {
        "sentences": sentences,
        "token_id_seqs": token_id_seqs,
        "freq_tables": freq_tables,
        "coverage": coverage,
        "vocab_overlap": vocab_overlap,
        "fertility_variance": fertility_variance
    }
