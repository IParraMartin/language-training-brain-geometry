"""Tests whether stimulus token frequency explains the brain-score gap between corpora.

Loads each model's best-layer brain score from results/, builds per-corpus token-frequency tables, optionally
computes embedding trainedness, and writes the cross-corpus regression to
results/confounds/correlational_analysis.json (or --out_path).

Example: brain-correlate --benchmark pereira_243 --corpus_model_ids python=IParraMartin/gpt2-python
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from brain_code.config import DATASETS, EXPERIMENTS, PATHS
from brain_code.diagnostics.token_confounds import (
    build_corpus_freq_table,
    tokenize_stimuli
)
from brain_code.diagnostics.correlational_analysis import (
    compute_embedding_trainedness,
    correlational_analysis
)


def _corpus_model_pair(value: str) -> tuple[str, str]:
    """Parse and validate a ``'corpus=model_id'`` argument string.

    Args:
        value (str): A string of the form ``'corpus=model_id'``, where
            ``corpus`` is a key in ``brain_code.config.DATASETS`` and
            ``model_id`` is a HuggingFace Hub model identifier or local
            checkpoint path.

    Returns:
        tuple[str, str]: A ``(corpus, model_id)`` pair.

    Raises:
        argparse.ArgumentTypeError: If the string does not contain ``'='`` or
            if ``corpus`` is not a registered dataset key.
    """
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected 'corpus=model_id', got: {value!r}. "
            "Example: python=IParraMartin/gpt2-python"
        )
    corpus, model_id = value.split("=", 1)
    if corpus not in DATASETS:
        raise argparse.ArgumentTypeError(
            f"Unknown corpus {corpus!r}. "
            f"Registered corpora: {list(DATASETS.keys())}"
        )
    return corpus, model_id


def _load_brain_scores(
    corpus_model_map: dict[str, str],
    benchmark_key: str,
    results_dir: Path
) -> dict[str, dict]:
    """Load per-model brain score JSON files and extract overall scores.

    Result files are matched on their contents, not their names: the harness
    names each result by a hash, so every JSON in ``results_dir`` is read and
    matched on ``model_identifier`` + ``benchmark_identifier``. When a model
    has several results (e.g. a descriptive grid and a single-layer run), the
    most recently written one wins.

    Args:
        corpus_model_map (dict[str, str]): Mapping from corpus name to
            HuggingFace model ID, e.g. ``{'python': 'IParraMartin/gpt2-python'}``.
        benchmark_key (str): A key from ``brain_code.config.EXPERIMENTS``
            (e.g. ``'pereira_243'``), resolved to the benchmark identifier
            recorded in each result file.
        results_dir (Path): Directory that contains the brain score JSON files.

    Returns:
        dict[str, dict]: Mapping from corpus name to a dict with key
            ``'overall'`` (float, best-layer score). Corpora with no matching
            result file, or none containing valid scores, are skipped with a
            warning.
    """
    benchmark_id = EXPERIMENTS[benchmark_key]
    brain_scores: dict[str, dict] = {}

    # model_identifier -> (result, mtime)
    by_model: dict[str, tuple[dict, float]] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("benchmark_identifier") != benchmark_id:
            continue
        model_identifier = payload.get("model_identifier")
        if not model_identifier:
            continue
        mtime = path.stat().st_mtime
        if model_identifier not in by_model or mtime > by_model[model_identifier][1]:
            by_model[model_identifier] = (payload, mtime)

    for corpus, model_id in corpus_model_map.items():
        if model_id not in by_model:
            print(
                f"  [WARNING] No {benchmark_id} result for {model_id} in "
                f"{results_dir} - skipping '{corpus}'"
            )
            continue
        result = by_model[model_id][0]
        rlr = result.get("raw_layer_results", {})
        valid = [
            v for v in rlr.values()
            if isinstance(v, dict) and "official_brainscore_raw_pearson" in v
        ]
        if not valid:
            print(f"  [WARNING] No valid scores for {model_id} - skipping '{corpus}'")
            continue
        best = max(valid, key=lambda x: x["official_brainscore_raw_pearson"])
        overall = best["official_brainscore_raw_pearson"]
        brain_scores[corpus] = {"overall": overall}
        print(f"  {corpus:<14}  best layer = {overall:.4f}  ({len(valid)} layers)")

    return brain_scores


def parse_args() -> argparse.Namespace:
    """Build and return the argument parser for the correlational analysis CLI.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        description=(
            "Run correlational analysis linking embedding trainedness to brain "
            "scores across training corpora."
        )
    )
    p.add_argument(
        "--corpus_model_ids",
        nargs="+",
        type=_corpus_model_pair,
        required=True,
        metavar="CORPUS=MODEL_ID",
        help=(
            "One or more 'corpus=model_id' pairs mapping each corpus to its "
            "trained model on HuggingFace Hub. "
            "Example: python=IParraMartin/gpt2-python tinystories=IParraMartin/gpt2-tinystories"
        )
    )
    p.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help=(
            "Benchmark that produced the brain scores; its stimuli are "
            "tokenized for the frequency analysis. "
            f"Available options: {', '.join(EXPERIMENTS.keys())}."
        )
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory containing brain score JSON files (default: results/)."
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="default",
        help=(
            "Tokenizer key for stimulus tokenization and frequency table "
            f"construction. Options: {list(PATHS['tokenizer'].keys())}. "
            "Defaults to 'default'."
        )
    )
    p.add_argument(
        "--text_field",
        type=str,
        default="code",
        help="Dataset column containing training text for frequency tables (default: 'code')."
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size for corpus frequency table construction (default: 4096)."
    )
    p.add_argument(
        "--compute_trainedness",
        action="store_true",
        help=(
            "Compute per-model embedding trainedness diagnostics before the "
            "correlational analysis. Each model is downloaded from the Hub. "
            "Skipped by default."
        )
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace access token for private or gated models."
    )
    p.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help=(
            "Output JSON path for correlational analysis results. "
            "Defaults to results/confounds/correlational_analysis.json."
        )
    )
    p.add_argument(
        "--init_embeddings",
        type=str,
        default=None,
        help=(
            "Random embeddings to get the effect of training more accurately. "
            "Defaults to `None`"
        )
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the brain-correlate CLI.

    Loads brain scores, tokenizes stimuli, builds per-corpus token-frequency
    tables, optionally computes per-model embedding trainedness, and runs the
    cross-corpus regression that tests whether stimulus token frequency
    confounds the brain-score gap.
    """
    args = parse_args()

    corpus_model_map: dict[str, str] = dict(args.corpus_model_ids)
    tokenizer_path: str = PATHS["tokenizer"][args.tokenizer]

    print("=" * 70)
    print("CORRELATIONAL ANALYSIS")
    print("=" * 70)

    print("\n[Step 1] Loading brain scores...")
    brain_scores = _load_brain_scores(
        corpus_model_map=corpus_model_map,
        benchmark_key=args.benchmark,
        results_dir=args.results_dir
    )
    if not brain_scores:
        raise RuntimeError(
            "No brain scores could be loaded. "
            "Check --results_dir and --corpus_model_ids."
        )

    print("\n[Step 2] Tokenizing stimuli...")
    sentences, token_id_seqs = tokenize_stimuli(
        benchmark_identifier=args.benchmark,
        tokenizer_name=tokenizer_path
    )
    print(f"  {len(sentences)} stimuli tokenized.")

    print("\n[Step 3] Building corpus frequency tables...")
    freq_tables: dict[str, Counter] = {}
    for corpus in corpus_model_map:
        if corpus not in brain_scores:
            continue
        print(f"  -> {corpus}")
        freq_tables[corpus] = build_corpus_freq_table(
            corpus_name=corpus,
            tokenizer_name=tokenizer_path,
            text_field=args.text_field,
            batch_size=args.batch_size
        )

    if args.compute_trainedness:
        print("\n[Step 4] Computing embedding trainedness per model...")
        for corpus, model_id in corpus_model_map.items():
            if corpus not in brain_scores:
                continue
            print(f"  -> {corpus}  ({model_id})")
            compute_embedding_trainedness(
                corpus_name=corpus,
                model_id=model_id,
                freq_table=freq_tables[corpus],
                stimulus_token_ids=token_id_seqs,
                stimulus_sentences=sentences,
                hf_token=args.hf_token,
                init_embeddings=args.init_embeddings
            )

    step = 5 if args.compute_trainedness else 4
    print(f"\n[Step {step}] Running correlational analysis...")
    correlational_analysis(
        brain_scores=brain_scores,
        freq_tables=freq_tables,
        stimulus_token_ids=token_id_seqs,
        out_path=args.out_path
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
