"""Measures tokenizer fertility and, optionally, per-corpus token counts.

Writes results/fertility_results.json and, with --corpora_metrics, results/token_counts.json.

Example: brain-measure --tokenizer default --corpora_metrics
"""

from __future__ import annotations

import argparse

from brain_code.config import PATHS
from brain_code.diagnostics.measurement_tools import (
    run_tokenizer_metrics,
    count_corpus_tokens
)


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokenizer",
        type=lambda k: PATHS["tokenizer"][k],
        default="default",
        help="Tokenizer key from PATHS config (default: 'default')."
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Number of tokens to process per batch (default: 4096)."
    )
    p.add_argument(
        "--corpora_metrics",
        action="store_true",
        help="If set, also compute per-corpus token counts."
    )
    return p.parse_args()


def main() -> None:
    """Computes the tokenizer metrics, then the corpus counts when requested."""
    args = parse_args()

    print("Computing tokenizer metrics...")
    run_tokenizer_metrics(
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size
    )

    if args.corpora_metrics:
        print("\n=====================\n")
        print("Computing corpora metrics...")
        count_corpus_tokens(
            tokenizer_name=args.tokenizer,
            batch_size=args.batch_size
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
