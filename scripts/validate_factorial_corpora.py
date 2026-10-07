"""Validates a built factorial corpus set.

Runs the corpus validation battery on condition_{A,B,C,D}.txt in the corpus
directory, prints per-check pass/fail, and exits 1 if any check fails.

Example: brain-validate-corpora --corpus_dir data/factorial_corpora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brain_code.training.corpus_validation import run_validation_battery, all_passed


def main() -> None:
    p = argparse.ArgumentParser(description="Validate factorial corpus conditions.")
    p.add_argument(
        "--corpus-dir",
        "--corpus_dir",
        dest="corpus_dir",
        required=True,
        help="Directory produced by brain-build-corpora."
    )
    args = p.parse_args()

    corpus_dir = Path(args.corpus_dir)

    def _read(cond: str) -> list[str] | None:
        path = corpus_dir / f"condition_{cond}.txt"
        if not path.exists():
            return None
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    texts_a = _read("A")
    if texts_a is None:
        sys.exit(f"condition_A.txt not found in {corpus_dir}")

    results = run_validation_battery(
        texts_a=texts_a,
        texts_b=_read("B"),
        texts_c=_read("C"),
        texts_d=_read("D"),
        verbose=True
    )
    sys.exit(0 if all_passed(results) else 1)


if __name__ == "__main__":
    main()
