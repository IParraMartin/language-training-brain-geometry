"""Runs the shared-tokenizer fairness diagnostics for one benchmark.

Builds per-corpus token frequency tables, measures stimulus token coverage and
reports cross-corpus vocabulary overlap, writing everything under
results/confounds/.

Example: brain-confounds --benchmark_id Pereira2018.243sentences-ridge
"""

from __future__ import annotations

import argparse

from brain_code.diagnostics.token_confounds import run_all_diagnostics


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Run shared-tokenizer fairness diagnostics: build corpus frequency "
            "tables, compute stimulus token coverage, and "
            "report cross-corpus vocabulary overlap."
        )
    )
    p.add_argument(
        "--benchmark_id",
        type=str,
        required=True,
        help="Identifier of the benchmark to run token confound diagnostics on."
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Number of tokens to process per batch (default: 4096)."
    )
    return p.parse_args()


def main() -> None:
    """Runs every diagnostic for the requested benchmark."""
    args = parse_args()
    print(f"Running analysis on {args.benchmark_id}...")
    run_all_diagnostics(
        benchmark_id=args.benchmark_id,
        corpora=None,
        batch_size=args.batch_size
    )
    print("Done.")


if __name__ == "__main__":
    main()
